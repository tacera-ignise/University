#pragma once
#include <functional>
#include <stdexcept>
#include <iostream>
#include <string>
#include <time.h>
#include <iomanip>
#include <sstream>
#include <Windows.h>
#include <random>

using namespace std;

template<typename T>
class Table {
private:
    T* data;
    int size;
    int space;

public:
    Table() {
        size = 0;
        space = 1;
        data = new T[space];
    }
    Table( T* initialValue, int initialSize) {
        size = 0;
        space = initialSize;
        data = new T[space];
        for (int i = 0; i < initialSize; ++i) {
            data[i] = initialValue[i];
        }
        size = initialSize;
    }

    ~Table() {
        clear();
    }

    int getSize() const { return size; }
    int setSize(int i) { return size = i; }
    void putData(T dane, int index) { data[index] = dane; }

    T& operator[](int index) { return data[index]; }

    const T& operator[](int index) const { return data[index]; }

    int comp(T n_1, T n_2) {
        if (n_1 > n_2) return 1;
        else if (n_1 < n_2) return -1;
        else return 0;
    }

    void add(T dane) {
        if (space == size) {
            space = space * 2;
            T* temp_array = new T[space];
            for (int i = 0; i < size; i++)
                temp_array[i] = data[i];
            delete[] data;
            data = temp_array;
        }
        data[size] = dane;
        size++;
    }

    void clear() {
        delete[] data;
        size = 0;
        space = 1;
        data = new T[space];
    }
};

template <typename T, typename Info = less<T>>
class Copiec {
private:
    Table<T> heap;
    Info comparator;
    bool direction{};

    void heapifyUp(int index) {
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

    void heapifyDown(int index, int size) {
        while (true) {
            int leftChild = 2 * index + 1;
            int rightChild = 2 * index + 2;
            int big = index;

            if (leftChild < size && comparator(heap[big], heap[leftChild]))
                big = leftChild;

            if (rightChild < size && comparator(heap[big], heap[rightChild]))
                big = rightChild;

            if (big != index) {
                swap(heap[index], heap[big]);
                index = big;
            }
            else {
                break;
            }
        }
    }

    void buildHeap() {
        int size = heap.getSize();
        for (int i = size / 2 - 1; i >= 0; --i) {
            heapifyDown(i, size);
        }
    }
public:
    Copiec(T* injectedArray, int size)
        : heap(injectedArray, size), comparator() {
        buildHeap();
    }
    

    Copiec() : comparator(Info()) {}
    void sort() {
        int size = heap.getSize();
        buildHeap();
        for (int i = size - 1; i > 0; --i) {
            swap(heap[0], heap[i]);
            heapifyDown(0, i);
        }
    }

    void insert(const T& data) {
        heap.add(data);
        heapifyUp(heap.getSize() - 1);
    }

    T poll() {
        if (heap.getSize() == 0) {
            throw out_of_range("Heap is empty");
        }

        T maxElement = heap[0];
        heap[0] = heap[heap.getSize() - 1];
        heap.setSize(heap.getSize() - 1);
        heapifyDown(0);

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
