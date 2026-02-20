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
    ~Table() {

        while (size == 0) {
            data[size - 1] = NULL;
            size--;
        }
        delete data;
    }
    int getSize() { return size; }
    void putData(T dane, int index) { data[index] = dane; }
    T& operator[](int index) {
        return data[index];
    }
    int comp(T n_1, T n_2) {
        if (n_1 > n_2)return 1;
        else if (n_1 < n_2)return -1;
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
        while (size == 0) {
            delete data[size - 1];
            data[size - 1] = nullptr;
            size--;
        }
        delete data;

    }
    void bubble_sort()
    {
        for (size_t i = 0; i < size - 1; i++)
        {
            for (size_t j = 0; j < size - i - 1; j++)
            {
                int cmp_result = comp(data[j], data[j + 1]);
                if (cmp_result == 1)
                {
                    swap(data[j], data[j + 1]);
                }
            }
        }
    }
};