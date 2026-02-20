struct Edge {
    int src, dest, weight;
};

template<typename T>
class Table {
private:
    T* data;
    int size;
    int space;

public:
    Table() : size(0), space(1), data(new T[space]) {}

    Table(int n) : size(n), space(n), data(new T[space]) {}

    ~Table() {
        delete[] data;
    }

    int getSize() const { return size; }

    void putData(T dane, int index) { data[index] = dane; }

    T& operator[](int index) {
        return data[index];
    }

    const T& operator[](int index) const {
        return data[index];
    }

    int comp(const T& n_1, const T& n_2, bool (*comparer)(const T&, const T&)) {
        return comparer(n_1, n_2);
    }

    void add(T dane) {
        if (size == space) {
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
        while (size > 0) {
            size--;
            data[size] = T();
        }
    }

    void bubble_sort(bool (*comparer)(const T&, const T&)) {
        for (int i = 0; i < size - 1; i++) {
            for (int j = 0; j < size - i - 1; j++) {
                if (comp(data[j], data[j + 1], comparer) > 0) {
                    std::swap(data[j], data[j + 1]);
                }
            }
        }
    }

private:
    int partition(bool (*comparer)(const T&, const T&), int low, int high) {
        T pivot = data[high];
        int i = (low - 1);

        for (int j = low; j <= high - 1; j++) {
            if (comp(data[j], pivot, comparer) > 0) {
                i++;
                std::swap(data[i], data[j]);
            }
        }
        std::swap(data[i + 1], data[high]);
        return (i + 1);
    }
};