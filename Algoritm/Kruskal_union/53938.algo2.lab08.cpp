//ALGO2 LS1 210A LAB08
//Daniil Protsvitaiev
//pd53938@zut.edu.pl
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
//1 cortirowka po wagam. 2. lączenie
using namespace std;

struct Node {
    double x, y;
};

struct Edge {
    int src, dest;
    double weight;
};

class UnionFind {
public:
    UnionFind(int n);
    int find(int x);
    void unionSets(int x, int y);

private:
    vector<int> parent;
    vector<int> rank;
};

UnionFind::UnionFind(int n) : parent(n), rank(n, 0) {
    for (int i = 0; i < n; ++i) {
        parent[i] = i;
    }
}

int UnionFind::find(int x) {
    if (parent[x] != x) {
        parent[x] = find(parent[x]);  
    }
    return parent[x];
}

void UnionFind::unionSets(int x, int y) {
    int rootX = find(x);
    int rootY = find(y);

    if (rootX != rootY) {
        if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
        }
        else if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
        }
        else {
            parent[rootX] = rootY;
            rank[rootY]++;
        }
    }
}

vector<Edge> kruskal(vector<Edge>& edges, int numNodes) {
    UnionFind uf(numNodes);
    vector<Edge> mst;

    for (const Edge& edge : edges) {
        int rootSrc = uf.find(edge.src);
        int rootDest = uf.find(edge.dest);

        if (rootSrc != rootDest) {
            mst.push_back(edge);
            uf.unionSets(edge.src, edge.dest);
        }
    }

    return mst;
}
int main() {
    string filenames[] = { "g1.txt", "g2.txt", "g3.txt" };

    for (const string& filename : filenames) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Unable to open file " << filename << endl;
            continue;
        }

        int numNodes;
        file >> numNodes;

        vector<Node> nodes(numNodes);
        for (int i = 0; i < numNodes; ++i) {
            file >> nodes[i].x >> nodes[i].y;
        }

        int numEdges;
        file >> numEdges;

        vector<Edge> edges(numEdges);
        for (int i = 0; i < numEdges; ++i) {
            file >> edges[i].src >> edges[i].dest >> edges[i].weight;
        }

        file.close();

        // Pomiar czasu sortowania krawędzi
        auto startSort = chrono::high_resolution_clock::now();
        sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
            return a.weight < b.weight;
            });
        auto stopSort = chrono::high_resolution_clock::now();
        auto durationSort = chrono::duration_cast<chrono::microseconds>(stopSort - startSort);

        auto startKruskal = chrono::high_resolution_clock::now();
        vector<Edge> mst = kruskal(edges, numNodes);
        auto stopKruskal = chrono::high_resolution_clock::now();
        auto durationKruskal = chrono::duration_cast<chrono::microseconds>(stopKruskal - startKruskal);

        cout << "Results for " << filename << ":" << endl;
        cout << "Number of edges in MST: " << mst.size() << endl;
        double sumOfWeights = 0.0;
        for (const Edge& edge : mst) {
            sumOfWeights += edge.weight;
        }
        cout << "Sum of weights in MST: " << sumOfWeights << endl;
        cout << "Time for sorting edges: " << durationSort.count() << " microseconds" << endl;
        cout << "Time for Kruskal's algorithm: " << durationKruskal.count() << " microseconds" << endl;
    }

    return 0;
}


