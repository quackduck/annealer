#include <iostream>
#include <vector>
#include <random>
#include <thread>

using namespace std;

using solution_t = vector<bool>;
using qubo_t = vector<vector<double>>;

struct settings {
    int max_iter;
    double T_0;
    double (*temp_scheduler)(double T_0, double T, int iter, int max_iter);
    unsigned seed;
};

struct result {
    solution_t solution;
    double energy;
};

double evaluate(const solution_t& x, const qubo_t& Q) {
    double value = 0.0;
    for (int i = 0; i < x.size(); i++) 
        for (int j = 0; j < x.size(); j++)
            value += Q[i][j] * x[i] * x[j];
    return value;
}

result sim_anneal(const qubo_t& Q, const settings s) { // intentionally get copy of settings
    mt19937 gen(s.seed);
    uniform_real_distribution<> dis(0.0, 1.0);
    uniform_int_distribution<> flip_dis(0, Q.size() - 1);

    solution_t x(Q.size(), 0);
    for (auto xi : x) {
        xi = dis(gen) < 0.5 ? 0 : 1;
    }
    double f_x = evaluate(x, Q);

    solution_t best_x = x;
    double best_f_x = f_x;

    double T = s.T_0;
    for (int iter = 0; iter < s.max_iter; iter++) {
        solution_t x_prime = x;
        int flip_idx = flip_dis(gen);
        x_prime[flip_idx] = !x_prime[flip_idx];

        double f_x_prime = evaluate(x_prime, Q);

        if (f_x_prime < f_x || dis(gen) < exp((f_x - f_x_prime) / T)) {
            x = x_prime;
            f_x = f_x_prime;
        }

        if (f_x < best_f_x) {
            best_x = x;
            best_f_x = f_x;
            // cout << "New best energy: " << best_f_x << endl;
            // cout << "New best solution: ";
            // for (auto xi : best_x) {
            //     cout << xi << " ";
            // }
        }

        T = s.temp_scheduler(s.T_0, T, iter, s.max_iter);
    }

    // cout << "Best energy: " << best_f_x << endl;

    return {best_x, best_f_x};
}

result multithreaded_sim_anneal(const qubo_t& Q, const settings s, int num_threads) {
    vector<thread> threads;
    vector<result> results(num_threads);
    for (int i = 0; i < num_threads; i++) {
        settings s_copy = s;
        s_copy.seed += i;
        threads.emplace_back([&results, i, &Q, s_copy](){
            results[i] = sim_anneal(Q, s_copy);
        });
    }

    for (auto& t : threads) t.join();

    return *min_element(results.begin(), results.end(), [](const result& a, const result& b) {
        return a.energy < b.energy;
    });
}

double linear_scheduler(double T_0, double T, int iter, int max_iter) {
    return T_0 - (T_0 / max_iter) * iter;
}

double geometric_scheduler(double T_0, double T, int iter, int max_iter) {
    return T * 0.99;
}

ostream& operator << (ostream& os, const solution_t& x) {
    for (auto xi : x)
        os << xi << ' ';
    return os;
}

void trial(solution_t x, const qubo_t& Q) {
    cout << "For solution: " << x << endl;
    cout << "Energy: " << evaluate(x, Q) << endl << endl;
}

int main() {
    // qubo_t Q = {
    //     { -2,  1,  1,  0,  0 },
    //     {  1, -2,  0,  1,  0 },
    //     {  1,  0, -3,  1,  1 },
    //     {  0,  1,  1, -3,  1 },
    //     {  0,  0,  1,  1, -2 }
    // };

    qubo_t Q = {
        {-17, 10, 10, 10, 0, 20},
        {10, -18, 10, 10, 10, 20},
        {10, 10, -29, 10, 20, 20},
        {10, 10, 10, -19, 10, 10},
        {0, 10, 20, 10, -17, 10},
        {20, 20, 20, 10, 10, -28}
    };

    trial({1, 0, 0, 0, 1, 0}, Q);

    // unsigned seed = time(0); // 42
    random_device rd;
    unsigned seed = rd();

    settings s = {.max_iter = 1000, .T_0 = 100.0, .temp_scheduler = linear_scheduler, .seed = seed};

    result r = multithreaded_sim_anneal(Q, s, 4);

    cout << "Best energy: " << r.energy << endl;
    cout << "Best solution: " << r.solution << endl;
    return 0;
}