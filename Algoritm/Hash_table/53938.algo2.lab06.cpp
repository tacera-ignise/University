//ALGO2 LS1 210A LAB06
//Daniil Protsvitaiev
//pd53938@zut.edu.pl
#include"lista.h"

// ...


template <typename V>
class HashTable {
private:
    List<pair<string, V>>* table;
    int size;
    int capacity;
    const double loadFactor = 0.75;

    int hashFunction(const string& key);
    void rehash();

public:
    HashTable(int capacity);
    ~HashTable();

    void insert(const string& key, const V& value);
    pair<string, V>* search(const string& key);
    bool remove(const string& key);
    void clear();
    string toString();
    int calculateHash(const string& key);
    void rehashAndInsert(const string& key, const V& value);
    void calculateStatistics();
};

template <typename V>
HashTable<V>::HashTable(int capacity) {
    this->capacity = capacity;
    this->size = 0;
    this->table = new List<pair <string, V>>[capacity];
}

template <typename V>
HashTable<V>::~HashTable() {
    clear();
    delete[] table;
}

template <typename V>
int HashTable<V>::hashFunction(const string& key) {
    int hash = 0;
    for (char c : key) {
        hash = (hash * 31 + c) % capacity;
    }
    return hash;
}

template <typename V>
void HashTable<V>::rehash() {
    int newCapacity = capacity * 2;
    List<pair<string, V>>* newTable = new List<pair<string, V>>[newCapacity];

    for (int i = 0; i < capacity; ++i) {
        Node<pair<string, V>>* current = table[i].head;
        while (current != nullptr) {
            pair<string, V> keyValue = current->data;
            int newHash = hashFunction(keyValue.first) % newCapacity;
            newTable[newHash].push_back(keyValue);
            current = current->next;
        }
    }

    delete[] table;
    table = newTable;
    capacity = newCapacity;
}

template <typename V>
void HashTable<V>::insert(const string& key, const V& value) {
    if (size + 1 > loadFactor * capacity) {
        rehash();
    }

    int hash = hashFunction(key) % capacity;
    Node<pair<string, V>>* current = table[hash].head;

    while (current != nullptr) {
        if (current->data.first == key) {
            current->data.second = value;
            return;
        }
        current = current->next;
    }

    table[hash].push_back(make_pair(key, value));
    size++;
}

template <typename V>
pair<string, V>* HashTable<V>::search(const string& key) {
    int hash = hashFunction(key) % capacity;
    Node<pair<string, V>>* current = table[hash].head;

    while (current != nullptr) {
        if (current->data.first == key) {
            return &(current->data);
        }
        current = current->next;
    }

    return nullptr;
}

template <typename V>
bool HashTable<V>::remove(const string& key) {
    int hash = hashFunction(key) % capacity;
    Node<pair<string, V>>* current = table[hash].head;

    while (current != nullptr) {
        if (current->data.first == key) {
            table[hash].erase(current);
            size--;
            return true;
        }
        current = current->next;
    }

    return false;
}

template <typename V>
void HashTable<V>::clear() {
    for (int i = 0; i < capacity; ++i) {
        table[i].clear();
    }
    size = 0;
}

template <typename V>
string HashTable<V>::toString() {
    string result = "";
    for (int i = 0; i < capacity; ++i) {
        Node<pair<string, V>>* current = table[i].head;
        while (current != nullptr) {
            result += "(" + current->data.first + ", " + to_string(current->data.second) + ") ";
            current = current->next;
        }
        if (table[i].Get_size() > 0) {
            result += "\n";
        }
    }
    return result;
}

template <typename V>
int HashTable<V>::calculateHash(const string& key) {
    return hashFunction(key) % capacity;
}

template <typename V>
void HashTable<V>::rehashAndInsert(const string& key, const V& value) {
    rehash();
    insert(key, value);
}

template <typename V>
void HashTable<V>::calculateStatistics() {
    int nonEmptyLists = 0;
    int minLength = numeric_limits<int>::max();
    int maxLength = 0;
    int totalLength = 0;

    for (int i = 0; i < capacity; ++i) {
        int length = table[i].Get_size();
        if (length > 0) {
            nonEmptyLists++;
            totalLength += length;

            if (length < minLength) {
                minLength = length;
            }

            if (length > maxLength) {
                maxLength = length;
            }
        }
    }

    cout << "Number of non-empty lists: " << nonEmptyLists << endl;
    cout << "Minimum list length: " << (minLength == numeric_limits<int>::max() ? 0 : minLength) <<endl;
    cout << "Maximum list length: " << maxLength << endl;
    cout << "Average list length: " << (nonEmptyLists > 0 ? static_cast<double>(totalLength) / nonEmptyLists : 0) << endl;
}

int main() {
    HashTable<int> hashTable(20);

    hashTable.insert("one", 1);
    hashTable.insert("two", 2);
    hashTable.insert("three", 3);
    hashTable.insert("four", 4);
    hashTable.insert("five", 5);
    hashTable.insert("six", 6);
    hashTable.insert("seven", 7);
    hashTable.insert("eight", 8);
    hashTable.insert("nine", 9);
    hashTable.insert("ten", 10);
    hashTable.insert("qw", 6);
    hashTable.insert("er", 7);
    hashTable.insert("ty", 8);
    hashTable.insert("ui", 9);
    hashTable.insert("yt", 10);
    hashTable.insert("six", 6);
    hashTable.insert("df", 7);
    hashTable.insert("gh", 8);
    hashTable.insert("jk", 9);
    hashTable.insert("zx", 10);

    cout << "HashTable content:\n" << hashTable.toString() << endl;

    pair<string, int>* result = hashTable.search("five");
    if (result != nullptr) {
        cout << "Found: (" << hashTable.search("five")->first << ", " << hashTable.search("five")->first << ")\n";
    }
    else {
        cout << "Not found.\n";
    }
    hashTable.rehashAndInsert("wer", 23);
    cout << "HashTable content:\n" << hashTable.toString() << endl;
    hashTable.remove("three");
    cout << "HashTable after removing 'three':\n" << hashTable.toString() << endl;
    
    hashTable.calculateStatistics();

    return 0;
}
