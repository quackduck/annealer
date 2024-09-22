#include <iostream>
#include <ostream>
#include <vector>
#include <random>
#include <thread>
#include "parser.hh"

using namespace std;

using solution_t = vector<bool>;
// using qubo_t = vector<vector<double>>;
using qubo_t = map<pair<int, int>, double>;

ostream& operator << (ostream& os, const solution_t& x) {
    for (auto xi : x) os << xi << ' ';
    return os;
}

ostream& operator << (ostream& os, const qubo_t& Q) {
    for (const auto& entry : Q) {
        os << '[' + entry.first.first << ' ' << entry.first.second << "] : " << entry.second << endl;
    }
    return os;
}

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
    for (const auto& entry : Q) {
        // value += entry.second * x[entry.first.first] * x[entry.first.second];
        if (x[entry.first.first] && x[entry.first.second]) value += entry.second;
    }
    return value;
}

int qubo_size(const qubo_t& Q) {
    int n = 0;
    for (const auto& entry : Q) {
        n = max(n, entry.first.first);
    }
    return n + 1; // 0 indexed
}

result sim_anneal(const qubo_t& Q, const settings s) { // intentionally get copy of settings
    int n = qubo_size(Q);

    mt19937 gen(s.seed);
    uniform_real_distribution<> dis(0.0, 1.0);
    uniform_int_distribution<> flip_dis(0, n - 1);

    solution_t x(n, 0);
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
    return T * 0.999;
}

void trial(solution_t x, const qubo_t& Q) {
    cout << "For solution: " << x << endl;
    cout << "Energy: " << evaluate(x, Q) << endl << endl;
}

// qubo_t unsparse(const map<pair<int, int>, double>& sparse) {
//     int n = 0;
//     for (const auto& entry : sparse) {
//         n = max(n, max(entry.first.first, entry.first.second));
//     }
//     n++;

//     qubo_t Q(n, vector<double>(n, 0.0));
//     for (const auto& entry : sparse) {
//         Q[entry.first.first][entry.first.second] = entry.second;
//     }
//     return Q;
// }

/* todo:
use sparse qubo
maybe flip all bits in sequence instead of random ones?
optimize evaluate by getting diff caused by flipping one bit
*/

int main() {
    // qubo_t Q = {
    //     { -2,  1,  1,  0,  0 },
    //     {  1, -2,  0,  1,  0 },
    //     {  1,  0, -3,  1,  1 },
    //     {  0,  1,  1, -3,  1 },
    //     {  0,  0,  1,  1, -2 }
    // };

    qubo_t Q = parse_qubo(read_file("qubo.txt"));

    // qubo_t Q = {
    //     {-17, 10, 10, 10, 0, 20},
    //     {10, -18, 10, 10, 10, 20},
    //     {10, 10, -29, 10, 20, 20},
    //     {10, 10, 10, -19, 10, 10},
    //     {0, 10, 20, 10, -17, 10},
    //     {20, 20, 20, 10, 10, -28}
    // };

    // qubo_t Q = unsparse(sparse); // just for now.

    // cout << Q << endl;

    // trial({1, 0, 0, 0, 1, 0}, Q);

    // unsigned seed = time(0); // 42
    random_device rd;
    unsigned seed = rd();

    settings s = {.max_iter = 10000, .T_0 = 100.0, .temp_scheduler = geometric_scheduler, .seed = seed};

    result r = multithreaded_sim_anneal(Q, s, 4);

    cout << "Best energy: " << r.energy << endl;
    cout << "Best solution: " << r.solution << endl;
    return 0;
}