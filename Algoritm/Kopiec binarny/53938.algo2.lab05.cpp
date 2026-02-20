//ALGO2 LS1 210A LAB05
//Daniil Protsvitaiev
//pd53938@zut.edu.pl
#include "tab_dyn.h"
#include <iostream>
#include <functional>
#include <stdexcept>

template <typename T, typename Info = less<T>>
class Copiec {
private:
    Table<T> heap;
    Info comparator;

    void Heap_if_Up(int index) {
        while (index > 0) {
            int parent = (index - 1) / 2;
            if (comparator(heap[parent], heap[index])) {
                swap(heap[index], heap[parent]);
                index = parent;
            }
            else {
                break;
            }
        }
    }

    void Heap_if_Down(int index) {
        int size = heap.getSize();
        while (true) {
            int leftChild = 2 * index + 1;
            int rightChild = 2 * index + 2;
            int Big = index;

            if (leftChild < size && comparator(heap[Big], heap[leftChild]))
                Big = leftChild;

            if (rightChild < size && comparator(heap[Big], heap[rightChild]))
                Big = rightChild;

            if (Big != index) {
                swap(heap[index], heap[Big]);
                index = Big;
            }
            else {
                break;
            }
        }
    }

public:
    void insert(const T& data) {
        heap.add(data);
        Heap_if_Up(heap.getSize() - 1);
    }

    T poll() {
        if (heap.getSize() == 0) {
            throw out_of_range("Heap is empty");
        }

        T maxElement = heap[0];
        heap[0] = heap[heap.getSize() - 1];
        heap.setSize(heap.getSize() - 1);  
        Heap_if_Down(0);

        return maxElement;
    }


    void clear() {
        heap.clear();
    }

    string toString() const {
        string result = "[";
        for (int i = 0; i < heap.getSize(); ++i) {
            result += to_string(heap[i]) + " ";
        }
        if (heap.getSize() > 0) {
            result.pop_back(); 
        }
        result += "]";
        return result;
    }
};

int main() {
    Copiec<int> cop;

    for (int i = 0; i < 20; i++) {
        cop.insert(rand() % 1000);
    }

    cout << "Heap: " << cop.toString() << endl;
    cout << "Heap: " << cop.toString() << endl;
    try {
        int maxElement = cop.poll();
        cout << "Extracted max element: " << maxElement << endl;
        cout << "Heap after extraction: " << cop.toString() << endl;
    }
    catch (const out_of_range& e) {
        cerr << e.what() << endl;
    }

    cop.clear();
    cout << "Heap after clearing: " << cop.toString() << endl;

    return 0;
}
