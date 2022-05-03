#include <stdio.h>
#include <math.h>
#include <vector>
#include "gurobi_c++.h"
#include <iostream>
#include <fstream>
using namespace std;



double eps_random(double eps, int i, int j, int omega) {
	i++; j++;

	double theta = 1 - omega / (omega - 1 + exp(eps));

	if (i == j)
		return theta + (1 - theta) / omega;
	else
		return (1 - theta) / omega;
}

double F(double mu, double b, double x) {
	if (x >= mu)
		return 1 - 0.5*exp(-(x - mu) / b);
	else
		return 0.5*exp((x - mu) / b);
}

double eps_laplace(double eps, int i, int j, int omega) {
	i++; j++;

	if (i == 1)
		return F(j, (omega - 1) / eps, 1.5);
	if (i == omega)
		return 1 - F(j, (omega - 1) / eps, omega - 0.5);

	return F(j, (omega - 1) / eps, i + 0.5) - F(j, (omega - 1) / eps, i - 0.5);
}


vector<double> es_pnum(vector<int> pnum, double eps, int omega, string LDP) {
	vector<double> ret(omega);

	for (int i = 0; i < omega; i++) {
		ret[i] = 0;
		for (int j = 0; j < omega; j++) {
			if (LDP == "random_response")
				ret[i] += pnum[j] * eps_random(eps, i, j, omega);
			if (LDP == "laplace")
				ret[i] += pnum[j] * eps_laplace(eps, i, j, omega);
		}
	}

	/*
	cout << LDP << endl;
	for (int i = 0; i < omega; i++)
		cout << ret[i] << " ";
	cout << endl;
	*/

	return ret;
}


vector<vector<double>> qmatrix(double eps, int omega, int n, vector<int> pnum, string LDP) {
	vector<double> lambda(1001);
	vector<vector<double>> q(1001);

	for (int i = 0; i < omega; i++)
		lambda[i] = pnum[i]*1.0 / n;

	for (int i = 0; i < omega; i++)
		q[i] = vector<double>(1001);

	for (int j = 0; j < omega; j++) {
		double cnt = 0;
		for (int k = 0; k < omega; k++) {
			if (LDP == "random_response")
				cnt += lambda[k] * eps_random(eps, j, k, omega);
			if (LDP == "laplace")
				cnt += lambda[k] * eps_laplace(eps, j, k, omega);
		}

		for (int i = 0; i < omega; i++) {
			if (LDP == "random_response")
				q[i][j] = (eps_random(eps, j, i, omega) * lambda[i]) / cnt;
			if (LDP == "laplace")
				q[i][j] = (eps_laplace(eps, j, i, omega) * lambda[i]) / cnt;
		}
	}
	return q;
}

double solve_PoLDP_plur(GRBEnv env, double eps, int n, int m, vector<int> pnum, vector<double> espnum, vector<vector<double>> qmatrix) {
	double delta = 0;
	//*sqrt(log(n) / n);

	GRBModel model = GRBModel(env);
	bool find_solution = false;

	GRBVar *x = new GRBVar[m];
	GRBVar *gamma = new GRBVar[m];
	

	/*
	cout << "-------" << endl;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			cout << qmatrix[i][j] << " ";
			//tmp += qmatrix[i][j];
		}
		cout << endl;
	}
	cout << "-------" << endl; 
	*/

	double opt = n;

	try {
		model.set(GRB_IntParam_OutputFlag, 0);

		for (int j = 0; j < m; j++) {
			x[j] = model.addVar(0.0, espnum[j], 0.0, GRB_CONTINUOUS);
			gamma[j] = model.addVar(0.0, n, 0.0, GRB_CONTINUOUS);
		}

		GRBLinExpr obj = 0;
		for (int j = 0; j < m; j++)
			obj += x[j];

		model.setObjective(obj, GRB_MINIMIZE);

		for (int i = 0; i < m; i++) {
				GRBLinExpr cons = pnum[i];
				for (int j = 0; j < m; j++) {
					if (i == 0)
						cons -= (1 + delta)*qmatrix[i][j] * x[j];
					else
						cons -= (1 - delta)*qmatrix[i][j] * x[j];
				}
				model.addConstr(gamma[i] == cons);
		}


		for (int i = 1; i < m; i++)
			model.addConstr(gamma[0] >= gamma[i]);

		model.optimize();

	
		if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
			opt = (model.get(GRB_DoubleAttr_ObjVal));
		}
	}
	catch (GRBException e) {
		cout << "Error code = " << e.getErrorCode() << endl;
		cout << e.getMessage() << endl;
	}
	catch (...) {
		cout << "Exception during optimization" << endl;
	}

	delete[] x;
	delete[] gamma;

	return opt;
}

int solve_exact_plur(int n, int m, vector<int> pnum) {
	int sum = 0; 
	for (int j = 1; j < m; j++)
		if (pnum[j] > pnum[0])
			sum += pnum[j] - pnum[0];
	return sum;
}

/*
void create_data_sushidata(int &n,int &m, vector<int> &pnum) {
	n = 5000;  m = 10;
	pnum[0] = 458;
	pnum[1] = 550;
	pnum[2] = 404;
	pnum[3] = 228;
	pnum[4] = 747;
	pnum[5] = 544;
	pnum[6] = 206;
	pnum[7] = 1713;
	pnum[8] = 113;
	pnum[9] = 37;

	n *= 200;
	for (int i = 0; i < m; i++)
		pnum[i] *= 200;
}
*/

void load_data_plurality(string file, int &instance_num, int &n, int &m, vector<vector<int>> &pnum) {
	ifstream fin(file);

	double phi;
	fin >> n >> m >> phi >> instance_num;

	pnum.clear();
	for (int i = 0; i < instance_num; i++) {
		vector<int> tmp_data(m);
		for (int j = 0; j < m; j++)
			fin >> tmp_data[j];
		pnum.push_back(tmp_data);
	}

	fin.close();
}


/*
void plot_plurality_sushidata() {
	string file = "plurality_sushidata.csv";
	ofstream fout(file);

	int n;
	int m;
	vector<int> pnum(1001);

	double eps_min = 0;
	double eps_max = 5;

	create_data_sushidata(n, m, pnum);

	GRBEnv env = GRBEnv();
	int opt = solve_exact_plur(n, m, pnum);

	for (double eps = eps_min; eps <= eps_max; eps += 0.01) {
		vector<vector<double>> qmatrix_rs = qmatrix(eps, m, n, pnum, "random_response");
		vector<vector<double>> qmatrix_lap = qmatrix(eps, m, n, pnum, "laplace");

		double opt_rs = solve_PoLDP_plur(env, eps, n, m, pnum, es_pnum(pnum, eps, m, "random_response"), qmatrix_rs);

		double opt_lap = solve_PoLDP_plur(env, eps, n, m, pnum, es_pnum(pnum, eps, m, "laplace"), qmatrix_lap);

		//cout << eps << "," << opt_rs*1.0 / opt << "," << opt_lap*1.0 / opt << endl;

		fout<<eps<<","<<opt_rs*1.0 / opt << "," << opt_lap*1.0 / opt << endl;
	}

	fout.close();
}
*/
double avg(vector<double> data) {
	double cnt = 0;
	for (int i = 0; i < data.size(); i++) cnt += data[i];
	return cnt / data.size();
}

double ma(vector<double> data) {
	double ret = data[0];
	for (int i = 0; i < data.size(); i++)
		if (ret < data[i]) ret = data[i];
	return ret;
}

double mi(vector<double> data) {
	double ret = data[0];
	for (int i = 0; i < data.size(); i++)
		if (ret > data[i]) ret = data[i];
	return ret;
}

double sd(vector<double> data) {
	double cnt = 0;
	double av = avg(data);
	for (int i = 0; i < data.size(); i++) cnt += (data[i] - av)*(data[i] - av);

	return sqrt(cnt / data.size());
}

void plot_plurality_randomdata(string file) {

	int instance_num;
	int n;
	int m;
	vector<vector<int>> pnum;

	load_data_plurality(file, instance_num, n, m, pnum);

	if (instance_num > 5000)
		instance_num = 5000;

	double eps_min = 0;
	double eps_max = 5;

	ofstream fout("plurality_randomdata_" + file + ".csv");

	GRBEnv env = GRBEnv();

	for (double eps = eps_min; eps <= eps_max; eps += 0.01) {
		
		int cnt_threshold_rs = 0;
		int cnt_threshold_lap = 0;

		vector<double> poldp_rs(instance_num);
		vector<double> poldp_lap(instance_num);

		for (int i = 0; i < instance_num; i++) {
			int opt = solve_exact_plur(n, m, pnum[i]);
			vector<vector<double>> qmatrix_rs = qmatrix(eps, m, n, pnum[i], "random_response");
			vector<vector<double>> qmatrix_lap = qmatrix(eps, m, n, pnum[i], "laplace");

			double opt_rs = solve_PoLDP_plur(env, eps, n, m, pnum[i], es_pnum(pnum[i], eps, m, "random_response"), qmatrix_rs);
			double opt_lap = solve_PoLDP_plur(env, eps, n, m, pnum[i], es_pnum(pnum[i], eps, m, "laplace"), qmatrix_lap);



			poldp_rs[i] = opt_rs*1.0 / opt;
			poldp_lap[i] = opt_lap*1.0 / opt;

			//if (poldp_rs[i] < 0.5)  cout << "xxxx" << endl;

			if (poldp_rs[i] >= 0.99 * n / opt) cnt_threshold_rs++;
			if (poldp_lap[i] >= 0.99 * n / opt) cnt_threshold_lap++;
		}

		//cout << cnt_threshold_lap << " " << cnt_threshold_lap*1.0 / instance_num << endl;

		cout << eps << "," << avg(poldp_rs) << "," << sd(poldp_rs) << "," << mi(poldp_rs) << "," << ma(poldp_rs) << "," << cnt_threshold_rs*1.0 / instance_num << ",";
		cout << avg(poldp_lap) << "," << sd(poldp_lap) << "," << mi(poldp_lap) << "," << ma(poldp_lap) << ","<<cnt_threshold_lap*1.0 / instance_num << endl;

		fout << eps << "," << avg(poldp_rs) << "," << sd(poldp_rs) << "," << mi(poldp_rs) << "," << ma(poldp_rs) << "," << cnt_threshold_rs*1.0 / instance_num << ",";
		fout << avg(poldp_lap) << "," << sd(poldp_lap) << "," << mi(poldp_lap) << "," << ma(poldp_lap) << ","<<cnt_threshold_lap*1.0 / instance_num << endl;

	}
	
	fout.close();
}



int factorial(int m) {
	int ret = 1;
	for (int i = 1; i <= m; i++)
		ret *= i;
	return ret;
}
void swap(int *a, int *b) { 
	*a ^= *b;
	*b ^= *a;
	*a ^= *b;
}
void permutation(vector<int> array, int k, vector<vector<int>> &ret) {
	if (k == array.size() - 1) {
		ret.push_back(array);
	}
	else {
		for (int i = k ; i < array.size(); i++) {
			swap(array[i], array[k]);
			permutation(array, k + 1, ret);
			swap(array[i], array[k]);
		}
	}
}
vector<vector<int>> generate_permutation(int m) {
	vector<vector<int>> ret;

	vector<int> tmp(m);

	for (int i = 0; i < m; i++)
		tmp[i] = i;

	permutation(tmp, 0, ret);

	return ret;

}
vector<int> generate_alpha(int m, string rule_name) {
	vector<int> ret(m);

	for (int i = 0; i < m; i++)
		ret[i] = 0;

	if (rule_name == "3app") {
		ret[0] = 1;
		ret[1] = 1;
		ret[2] = 1;
	}

	if (rule_name == "2app") {
		ret[0] = 1;
		ret[1] = 1;
	}

	if (rule_name == "bd") {
		for (int i = 0; i < m; i++)
			ret[i] = m - 1 - i;
	}

	return ret;
}
vector<vector<int>> scoring_matrix(vector<vector<int>> permutation, vector<int> alpha,int m) {
	vector<vector<int>> t;
	for (int i = 0; i < m; i++) {
		vector<int> tmp(factorial(m));

		for (int j = 0; j < permutation.size(); j++) {
			int k = 0;
			for (k = 0; k < m; k++)
				if (permutation[j][k] == i)
					break;
			tmp[j] = alpha[k];
		}
		t.push_back(tmp);
	}
	return t;
}


void load_data_score(string file, int &instance_num, int &n, int &m, vector<vector<int>> &pnum) {

	ifstream fin(file);

	double phi;
	fin >> n >> m >> phi >> instance_num;

	pnum.clear();
	for (int i = 0; i < instance_num; i++) {
		int omega = factorial(m);
		vector<int> tmp_data(omega);
		for (int j = 0; j < omega; j++)
			fin >> tmp_data[j];
		pnum.push_back(tmp_data);
	}

	fin.close();

}

double solve_exact_score(GRBEnv env, int n, int m, vector<int> pnum, string rule_name) {

	vector<int> alpha = generate_alpha(m, rule_name);

	vector<vector<int>> permu = generate_permutation(m);

	vector<vector<int>> t = scoring_matrix(permu, alpha, m);

	int omega = factorial(m);

	double delta = 0;

	GRBModel model = GRBModel(env);

	GRBVar *x = new GRBVar[omega];
	GRBVar *gamma = new GRBVar[omega];
	GRBVar *S = new GRBVar[m];
	double *cof = new double[omega];

	double opt = n;

	try {
		model.set(GRB_IntParam_OutputFlag, 0);

		for (int j = 0; j < omega; j++) {
			x[j] = model.addVar(0.0, pnum[j], 0.0, GRB_CONTINUOUS);
			gamma[j] = model.addVar(0.0, n, 0.0, GRB_CONTINUOUS);
		}

		for (int j = 0; j < m; j++) 
			S[j] = model.addVar(0.0, n * 100, 0.0, GRB_CONTINUOUS);


		GRBLinExpr obj = 0;
		for (int j = 0; j < omega; j++)
			obj += x[j];

		model.setObjective(obj, GRB_MINIMIZE);

		for (int j = 0; j < omega; j++)
			model.addConstr(gamma[j] == pnum[j] - x[j]);


		for (int i = 0; i < m; i++) {
			GRBLinExpr cons = 0;

			
			/////////
			for (int j = 0; j < omega; j++)
				cof[j] = t[i][j];
			cons.addTerms(cof, gamma, omega);
			/////////
			
			/*
			for (int j = 0; j < omega; j++)
				cons += gamma[j] * t[i][j];
			*/
			
			model.addConstr(S[i] == cons);
		}

		for (int i = 1; i < m; i++)
			model.addConstr(S[0] >= S[i]);

		model.optimize();


		if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
			opt = (model.get(GRB_DoubleAttr_ObjVal));
		}
	}
	catch (GRBException e) {
		cout << "Error code = " << e.getErrorCode() << endl;
		cout << e.getMessage() << endl;
	}
	catch (...) {
		cout << "Exception during optimization" << endl;
	}

	delete[] x;
	delete[] gamma;
	delete[] S;
	delete[] cof;

	return opt;
}

double solve_PoLDP_score(GRBEnv env, int n, int m, vector<int> pnum, string rule_name, vector<double> espnum, vector<vector<double>> qmatrix, string flag) {

	vector<int> alpha = generate_alpha(m, rule_name);

	vector<vector<int>> permu = generate_permutation(m);

	vector<vector<int>> t = scoring_matrix(permu, alpha, m);

	int omega = factorial(m);

	double delta = 0;

	if (flag == "up")  delta = 0.001;        //sqrt(log(n) / n);
	if (flag == "down")  delta = -0.001;
	if (flag == "norm")  delta = 0;

	GRBModel model = GRBModel(env);

	GRBVar *x = new GRBVar[omega];
	GRBVar *gamma = new GRBVar[omega];
	GRBVar *S = new GRBVar[m];
	double *cof = new double[omega];


	double opt = n;

	try {
		model.set(GRB_IntParam_OutputFlag, 0);

		for (int j = 0; j < omega; j++) {
			x[j] = model.addVar(0.0, espnum[j], 0.0, GRB_CONTINUOUS);
			gamma[j] = model.addVar(0.0, n, 0.0, GRB_CONTINUOUS);
		}

		for (int j = 0; j < m; j++)
			S[j] = model.addVar(0.0, n * 100, 0.0, GRB_CONTINUOUS);


		GRBLinExpr obj = 0;
		for (int j = 0; j < omega; j++)
			obj += x[j];


		model.setObjective(obj, GRB_MINIMIZE);

		for (int j = 0; j < omega; j++) {
			GRBLinExpr cons = pnum[j];


			///////
			for (int k = 0; k < omega; k++)
				cof[k] = -qmatrix[j][k];
			cons.addTerms(cof, x, omega);
			///////

			//for (int k = 0; k < omega; k++) {
			//		cons -=qmatrix[j][k] * x[k];
			//}
			model.addConstr(gamma[j] == cons);
		}


		for (int i = 0; i < m; i++) {
			GRBLinExpr cons = 0;

			///////
			for (int j = 0; j < omega; j++)
				cof[j] = t[i][j];
			cons.addTerms(cof, gamma, omega);
			//////

			//for (int j = 0; j < omega; j++)
			//	cons += gamma[j] * t[i][j];

			for (int j = 0; j < omega; j++) {
				//////
				for (int k = 0; k < omega; k++) {
					if (i == 0)
						cof[k] = -delta*(qmatrix[j][k] * t[i][j]);
					else
						cof[k] =  delta*(qmatrix[j][k] * t[i][j]);
				}
				cons.addTerms(cof, x, omega);
				//////

				/*
				for (int k = 0; k < omega; k++) {
					if (i == 0)
						cons -= delta*(x[k] * qmatrix[j][k] * t[i][j]);
					else
						cons += delta*(x[k] * qmatrix[j][k] * t[i][j]);
				}
				*/
			}

			model.addConstr(S[i] == cons);
		}


		for (int i = 1; i < m; i++)
			model.addConstr(S[0] >= S[i]);

		model.optimize();


		if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
			opt = (model.get(GRB_DoubleAttr_ObjVal));
		}
	}
	catch (GRBException e) {
		cout << "Error code = " << e.getErrorCode() << endl;
		cout << e.getMessage() << endl;
	}
	catch (...) {
		cout << "Exception during optimization" << endl;
	}

	delete[] x;
	delete[] gamma;
	delete[] S;
	delete[] cof;

	return opt;
}


void plot_score_randomdata(string file, string rule_name) {

	int instance_num;
	int n;
	int m;
	vector<vector<int>> pnum;

	load_data_score(file, instance_num, n, m, pnum);

	if (instance_num > 2000)
		instance_num = 2000;

	double eps_min = 0;
	double eps_max = 5;


	ofstream fout("score_randomdata_" + file + ".csv");

	GRBEnv env = GRBEnv();

	for (double eps = eps_min; eps <= eps_max; eps += 0.01) {

		int cnt_threshold_rs = 0;
		int cnt_threshold_lap = 0;

		vector<double> poldp_rs(instance_num);
		//vector<double> poldp_rs_delta(instance_num);

		vector<double> poldp_lap(instance_num);
		//vector<double> poldp_lap_delta(instance_num);

		for (int i = 0; i < instance_num; i++) {
			double opt = solve_exact_score(env, n, m, pnum[i], rule_name);
			
			vector<vector<double>> qmatrix_rs = qmatrix(eps, factorial(m), n, pnum[i], "random_response");
			vector<vector<double>> qmatrix_lap = qmatrix(eps, factorial(m), n, pnum[i], "laplace");
			
			double opt_rs;
			//double opt_rs_up;
			//double opt_rs_down;

			double opt_lap;
			//double opt_lap_up;
			//double opt_lap_down;


			{
				opt_rs = solve_PoLDP_score(env, n, m, pnum[i], rule_name, es_pnum(pnum[i], eps, factorial(m), "random_response"), qmatrix_rs, "normal");

				//opt_rs_up = solve_PoLDP_score(env, n, m, pnum[i], rule_name, es_pnum(pnum[i], eps, factorial(m), "random_response"), qmatrix_rs, "up");
				//opt_rs_down = solve_PoLDP_score(env, n, m, pnum[i], rule_name, es_pnum(pnum[i], eps, factorial(m), "random_response"), qmatrix_rs, "down");
			}

			{
				opt_lap = solve_PoLDP_score(env, n, m, pnum[i], rule_name, es_pnum(pnum[i], eps, factorial(m), "laplace"), qmatrix_lap, "normal");

				//opt_lap_up = solve_PoLDP_score(env, n, m, pnum[i], rule_name, es_pnum(pnum[i], eps, factorial(m), "laplace"), qmatrix_lap, "up");
				//opt_lap_down = solve_PoLDP_score(env, n, m, pnum[i], rule_name, es_pnum(pnum[i], eps, factorial(m), "laplace"), qmatrix_lap, "down");
			}



			poldp_rs[i] = opt_rs*1.0 / opt;
			poldp_lap[i] = opt_lap*1.0 / opt;

			//poldp_rs_delta[i] = (opt_rs_up - opt_rs_down) / opt_rs;
			//poldp_lap_delta[i] = (opt_lap_up - opt_lap_down) / opt_lap;


			if (poldp_rs[i] >= 0.99 * n / opt) cnt_threshold_rs++;
			if (poldp_lap[i] >= 0.99 * n / opt) cnt_threshold_lap++;
			
		}


		cout << eps << "," << avg(poldp_rs) << "," << sd(poldp_rs) << "," << mi(poldp_rs) << "," << ma(poldp_rs) << "," << cnt_threshold_rs*1.0 / instance_num << ",";
		//cout << avg(poldp_rs_delta) << "," << sd(poldp_rs_delta) << ",";
		cout << avg(poldp_lap) << "," << sd(poldp_lap) << "," << mi(poldp_lap) << "," << ma(poldp_lap) << "," << cnt_threshold_lap*1.0 / instance_num << endl;
		//cout << avg(poldp_lap_delta) << "," << sd(poldp_lap_delta) << endl;

		fout << eps << "," << avg(poldp_rs) << "," << sd(poldp_rs) << "," << mi(poldp_rs) << "," << ma(poldp_rs) << "," << cnt_threshold_rs*1.0 / instance_num << ",";
		//fout << avg(poldp_rs_delta) << "," << sd(poldp_rs_delta) << ",";
		fout << avg(poldp_lap) << "," << sd(poldp_lap) << "," << mi(poldp_lap) << "," << ma(poldp_lap) << "," << cnt_threshold_lap*1.0 / instance_num << endl;
		//fout << avg(poldp_lap_delta) << "," << sd(poldp_lap_delta) << endl;
		

	}

	fout.close();
}



int main() {


	plot_plurality_randomdata("2_0.100000.txt");
	plot_plurality_randomdata("2_0.200000.txt");
	plot_plurality_randomdata("2_0.300000.txt");
	plot_plurality_randomdata("2_0.400000.txt");

	
	plot_plurality_randomdata("5_0.100000.txt");
	plot_plurality_randomdata("5_0.200000.txt");
	plot_plurality_randomdata("5_0.300000.txt");
	plot_plurality_randomdata("5_0.400000.txt");
	

	plot_plurality_randomdata("plur_sushi_5.txt");


	
	plot_score_randomdata("3app_sushi_5.txt", "3app");
	plot_score_randomdata("bd_sushi_5.txt", "bd");

	plot_score_randomdata("3app_5_0.100000.txt", "3app");
	plot_score_randomdata("3app_5_0.200000.txt", "3app");

	plot_score_randomdata("bd_5_0.100000.txt", "bd");
	plot_score_randomdata("bd_5_0.200000.txt", "bd");


	//system("pause");
	return 0;
}