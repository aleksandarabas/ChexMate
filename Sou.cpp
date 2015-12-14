#include<iostream>
#include<algorithm>
#include<vector>
#include<string>
#include<cmath>
#include<fstream>
#include<cstring>
#include<iomanip>
using namespace std;


// Constants
double  gradient_alpha = 0.01;
#define discount 0.9999
#define black_king 3
#define white_king -3
#define black 2
#define white -2
#define empty_cell 0
#define num_layers 3
#define blackwin 1
#define whitewin 0
#define episode_length 500
#define R_tie 0.5
#define board_length 33
#define R 200
int Llayer[] = { 39 ,27, 1 , 1 };
int num_iterations;

//neural network
double sigmoid(double z) {
	return 1.0 / (1 + exp(-z));
}

vector< vector< vector < double> > > weights, tweights;
vector<double> layer[num_layers];
vector< double > delta[num_layers];
//Initialize Vectors
void init() {
	weights.resize(num_layers);
	tweights.resize(num_layers);
	for (int i = 0; i < num_layers; ++i)
	{
		layer[i].resize(Llayer[i] + 1);
		delta[i].resize(Llayer[i] + 1);
		if (i == 0) continue;
		weights[i].resize(Llayer[i] + 1);
		tweights[i].resize(Llayer[i] + 1);
		for (int j = 0;j < Llayer[i];++j)
		{
			weights[i][j].resize(Llayer[i - 1] + 1);
			tweights[i][j].resize(Llayer[i - 1] + 1);
			for (int k = 0; k < Llayer[i - 1] + 1; ++k)
				weights[i][j][k] = (2.0*rand() - RAND_MAX) / (RAND_MAX*2.0);
		}

	}
}
//Set all values in delta and tweights to 0
void zero_init() {
	for (int i = 0;i < num_layers;++i)
		for (int j = 0;j < delta[i].size();++j)
			delta[i][j] = 0;

	int Ni = weights.size();
	for (int i = 0;i < Ni;++i) {
		int Nj = weights[i].size();
		for (int j = 0;j < Nj;++j) {
			int Nk = weights[i][j].size();
			for (int k = 0; k < Nk;++k) tweights[i][j][k] = 0;
		}
	}

}
double Estimate(vector< double > &layer0)// Forward-Propgation
{


	layer[0] = layer0;
	for (int i = 1;i < num_layers;++i) {


		for (int j = 0; j < Llayer[i]; ++j) {

			layer[i][j] = 0;
			for (int k = 0; k < Llayer[i - 1];++k) {
				layer[i][j] += layer[i - 1][k] * weights[i][j][k];

			}
			layer[i][j] += weights[i][j][Llayer[i - 1]];// Bias Unit
			layer[i][j] = sigmoid(layer[i][j]);
		}
	}

	return layer[num_layers - 1][0];

}

void calc_gradient(vector<double>  &layer0, double &y) { // back-propgation, The gradient is saved in tweights
	zero_init();

	Estimate(layer0);

	delta[num_layers - 1][0] = (y - layer[num_layers - 1][0])*(1 - layer[num_layers - 1][0])*layer[num_layers - 1][0];

	for (int i = num_layers - 2; i > 0; --i) {
		for (int j = 0; j < Llayer[i];++j) {

			delta[i][j] = 0;
			for (int k = 0; k < Llayer[i + 1]; ++k)
				delta[i][j] += delta[i + 1][k] * weights[i + 1][k][j];
			delta[i][j] = layer[i][j] * (1 - layer[i][j]) * delta[i][j];
		}
	}

	for (int i = 1; i < num_layers;++i) {
		for (int j = 0; j < Llayer[i];++j) {
			for (int k = 0;k < Llayer[i - 1];++k)
			{
				tweights[i][j][k] += gradient_alpha*(delta[i][j] * layer[i - 1][k]);
			}
			tweights[i][j][Llayer[i - 1]] += gradient_alpha* delta[i][j];// Bias Unit
		}
	}


}
void iter(vector<double>  &layer0, double &y) {//Gradient descent
	calc_gradient(layer0, y);
	for (int i = 1; i < num_layers;++i) {
		for (int j = 0; j < Llayer[i];++j) {
			for (int k = 0;k <= Llayer[i - 1];++k)
			{
				weights[i][j][k] += tweights[i][j][k];

			}

		}
	}
}

/////// Checkers Environment
inline pair<int, int> P_to_XY(int p) { //Position in board vector to (X,Y) coordinates 
	pair<int, int> ret;
	ret.first = p / 4;
	ret.second = 2 * (p % 4) + 1 - (ret.first % 2);
	return ret;
}

inline int XY_to_P(pair<int, int > XY) {// (X,Y) coordinates to position in vector
	int ret = 4 * XY.first;
	ret += (XY.second - 1 + (XY.first) % 2) / 2;
	return ret;
}
int dx[] = { 1,1,-1,-1 };// the change in x and y in all 4 directions
int dy[] = { -1,1,-1,1 };
void dfs(int i, vector<double> cur_state, vector<vector<double> > &ret) {// add all possible next states that involve capturing pieces to ret
	pair<int, int> XY = P_to_XY(i);
	int x = XY.first;
	int y = XY.second;

	for (int j = 0;j < 4; ++j) {
		int NX = x + 2 * dx[j], NY = y + 2 * dy[j];
		int MX = x + dx[j], MY = y + dy[j];
		int P = XY_to_P(make_pair(NX, NY)),
			MP = XY_to_P(make_pair(MX, MY));

		if (NX < 0 || NX>7 || NY < 0 || NY> 7 || cur_state[P] != empty_cell) continue;
		if ((cur_state[MP] > 0 && cur_state[i] > 0) || (cur_state[MP] < 0 && cur_state[i] < 0) || cur_state[MP] == empty_cell)continue;

		if ((abs(cur_state[i]) == black_king)
			|| (j < 2 && cur_state[i] == black)
			|| (j >= 2 && cur_state[i] == white))
		{
			vector< double > tmp = cur_state;
			swap(tmp[i], tmp[P]);
			tmp[MP] = empty_cell;
			bool promo = 0;
			tmp[board_length - 1] = !tmp[board_length - 1];
			if (NX == 7 && tmp[P] == black) tmp[P] = black_king, promo = 1;
			if (NX == 0 && tmp[P] == white) tmp[P] = white_king, promo = 1;
			ret.push_back(tmp);
			tmp[board_length - 1] = !tmp[board_length - 1];
			if (!promo)dfs(XY_to_P(make_pair(NX, NY)), tmp, ret);
		}
	}
}
vector <vector<double>> Get_Next_States(vector< double > cur_state) { // Returns a vector that includes all next possible states
	vector<vector<double> > ret;

	for (int i = 0;i < board_length - 1; ++i)
		if ((cur_state[i]>0 && cur_state[board_length - 1] == 0) || (cur_state[i] < 0 && cur_state[board_length - 1] != 0))
		{
			dfs(i, cur_state, ret); // try to capture starting from position i
		}

	for (int i = 0;i < board_length - 1; ++i)
		if ((cur_state[i]>0 && cur_state[board_length - 1] == 0) || (cur_state[i]<0 && cur_state[board_length - 1] != 0))// try to move to directly adjacent square
		{
			pair< int, int > pos = P_to_XY(i);

			for (int j = 0;j < 4; ++j) {
				int NX = pos.first + dx[j], NY = pos.second + dy[j];
				int P = XY_to_P(make_pair(NX, NY));
				if (NX < 0 || NX>7 || NY < 0 || NY> 7 || cur_state[P] != empty_cell) continue;

				if ((abs(cur_state[i]) == black_king)
					|| (j < 2 && cur_state[i] == black)
					|| (j >= 2 && cur_state[i] == white))
				{
					vector< double > tmp = cur_state;
					swap(tmp[i], tmp[P]);
					// Promotion 
					if (NX == 7 && tmp[P] == black) tmp[P] = black_king;
					if (NX == 0 && tmp[P] == white) tmp[P] = white_king;
					tmp[board_length - 1] = !tmp[board_length - 1];
					ret.push_back(tmp);
				}
			}

		}
	return ret;
}

void dfs_mark(int i, vector<double> &cur_state, double sval, vector<int> &ret, vector<bool> &vis) {
	if (vis[i]) return;
	vis[i] = 1;
	pair<int, int> XY = P_to_XY(i);
	int x = XY.first;
	int y = XY.second;

	for (int j = 0;j < 4; ++j) {
		int NX = x + 2 * dx[j], NY = y + 2 * dy[j];
		int MX = x + dx[j], MY = y + dy[j];
		int P = XY_to_P(make_pair(NX, NY)),
			MP = XY_to_P(make_pair(MX, MY));

		if (NX < 0 || NX>7 || NY < 0 || NY> 7 || cur_state[P] != empty_cell) continue;
		if ((cur_state[MP] > 0 && sval> 0) || (cur_state[MP] < 0 && sval < 0) || cur_state[MP] == empty_cell)continue;

		if ((abs(sval) == black_king)
			|| (j < 2 && sval == black)
			|| (j >= 2 && sval == white))
		{

			ret[MP] = 1;

			dfs_mark(XY_to_P(make_pair(NX, NY)), cur_state, sval, ret, vis);
		}
	}
}
vector<int> mark_threatened(vector<double> V) {
	vector<int> ret;
	ret.resize(board_length);
	for (int i = 0;i < board_length-1;++i) {// mark all threatened with a capture sequence starting from i
		vector<bool> vis;
		vis.resize(board_length);
		if (V[i] != empty_cell)dfs_mark(i, V, V[i], ret, vis);
	}
	return ret;
}

vector <double> scale_to_layer(vector <double> &V) {// convert a board vector into a feature vector compatible with the neural network
	vector<double> layer1;
	layer1.resize(Llayer[0]);
	for (int i = 0;i < Llayer[0];++i)
		layer1[i] = 0;
	vector<int> threat = mark_threatened(V);

	for (int i = 0;i < board_length;++i) {
		layer1[0] += (V[i] >0);
		layer1[1] += (V[i] <0);
		layer1[2] += (V[i] == black_king);
		layer1[3] += (V[i] == white_king);
		layer1[4] += (threat[i] && V[i]>0);
		layer1[5] += (threat[i] && V[i] < 0);
		
	}
	
	for (int i = 0;i < board_length;++i)
		layer1[i + 6] = V[i];
	return layer1;
}
double g_val(vector<double> & V) {
	double x = Estimate(scale_to_layer(V));// gets estimation from neural network
	if (V[board_length - 1] == 0) x *= -1;//Multiplying to simplify code in play function, since one player is a maximizer and the other is a minimizer 
	return x;							// this way I can use a general maximization in the play function

}
bool chance(double E) {// Returns 1 with probability of E
	double R = rand();
	R /= RAND_MAX;
	return (R < E);
}
vector<double> rev(vector<double> state)// reverses the state..
{											// more concretely it flips the board and changes black pieces with their counterparts and viceversa  
	for (int i = 0, j = 31;i < j;++i, j--)swap(state[i], state[j]);// it generates the same state but with reverse colors
	for (int i = 0; i < board_length;++i)if(state[i]!=empty_cell) state[i] *= -1;
	state[board_length-1] = !state[board_length-1];
	return state;
}
vector<double> play(vector<double> state, bool E) { // if E =0 plays the best option based on neural network estimiation 
													// if E = 1 plays Randomly 

	vector<  vector<double> > next_states = Get_Next_States(state);
	int sz = next_states.size();
	
	if (E) {
		
		return next_states[rand() % sz];
	}
	
	
	int x = g_val(next_states[0]);
	pair<double, int> best = make_pair(g_val(next_states[0]), 0); // get the best next state 
	for (int i = 1;i < sz;++i) {
		best = max(best, make_pair(g_val(next_states[i]), i));
	}

	return next_states[best.second]; 


}

double reward(vector<double> state) {// returns reward for entering state as specified by constants

	vector<vector<double>> VV = Get_Next_States(state);
	if (VV.empty()) { // player will have no moves to play if all pieces are trapped or no piece is left both are losing states

		if (state[board_length - 1]) return  blackwin;
		return whitewin;
	}
	return R_tie; // if player can move the game goes on
}

vector<double> New_state() {// returns a fresh state with pieces in their starting positions 
	vector<double> ret;
	for (int i = 0;i < 12;++i)
		ret.push_back(black);
	for (int i = 0;i < 8;++i)
		ret.push_back(empty_cell);
	for (int i = 0;i < 12;++i)
		ret.push_back(white);
	ret.push_back(0);// black playes first
	return ret;
}

void iterations(vector<vector<double>> &layer1, vector<double> &y, int T) {// Trains current neural network by doing T gradient descent iterations
	int N = y.size();													// layer1 is the vector of states , y is the vector of expected output for states
	for (int i = 0;i < N;++i)
		layer1.push_back(rev(layer1[i])), y.push_back(1 - y[i]); // Train both players on the same state to reduce the chance of having a better player
	N += N;
	for (int i = 0;i < N;++i)
		layer1[i] = scale_to_layer(layer1[i]); // convert all states
	for (int i = 0;i < T;++i)
		iter(layer1[i%N], y[i%N]);// gradient descent iteration 
}
int wwin = 0, bwin = 0, draws = 0; // global veriables to keep track of current record of wins, loses and draws
int  obwin = 0, owwin = 0, odraws = 0;

void training_episode(double E) { // a training episode with probality E of playing randomly 

	vector<double> cur_state = New_state();
	
	vector<vector<double>> episode_states;// saves all states in current episode

	episode_states.push_back(cur_state);
	double last_reward = 0;
	
	for (int i = 0;i < episode_length;++i) {
		bool Z = chance(E);
		cur_state = play(cur_state, Z);

		last_reward = reward(cur_state);
		episode_states.push_back(cur_state);
		if (last_reward != R_tie) break;
	}
	if (last_reward == whitewin) { ++wwin, ++owwin; }
	else if (last_reward == blackwin) { ++bwin, ++obwin; }
	else ++draws;
	int N = episode_states.size();
	vector<double> Y;

	Y.resize(N);

	Y[N - 2] = Y[N - 1] = reward(episode_states[N - 1]);

	
	for (int i = N - 3;i >= 0;--i)
	{
		Y[i] = (Estimate(scale_to_layer(episode_states[i + 2]))); // generating Y values for dataset
		avg += Y[i];
	}

	iterations(episode_states, Y, num_iterations);
}

void train() {
	int N;
	cout << " Num of episodes: \n";
	cin >> N;
	cout << " Base num of iter: \n";
	cin >> num_iterations;
	double alpha = 1;
	int B =  num_iterations;
	
	for (int i = 0;i < N;++i) {
		num_iterations = B - B*double(i / N); // Number of iterations decreases over time
		gradient_alpha *= discount;	// Learning rate decreases overtime 
		double E = double(abs(obwin - owwin)) / R + 0.1; // computes E which is the propability of playing randomly in the next training episode
						
		training_episode(E);

		if (i % R==0) {
			obwin = 0, owwin = 0, odraws = 0;// reset values that influence E
		}
	
	}
}
int main(){
	init();
	train();// after the training is complete, the weights of the neural net will be   storted in the vector weights. 
	
}
