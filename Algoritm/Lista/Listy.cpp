//ALGO2 LS1 210A LAB01
//Daniil Protsvitaiev
//pd53938@zut.edu.pl
#include<time.h>
#include <iostream>

using namespace std;

template< typename T>

class Node {
public:
    Node* next;
    Node* prev;
    T data;
    Node(T data = T(), Node* next = nullptr, Node* prev = nullptr) {
        this->data = data;
        this->next = next;
        this->prev = prev;
    }
};

template<typename T>
class List {
public:
    Node<T>* head;
    Node<T>* tail;
    int Size;
    List();
    ~List();

    int Get_size() { return Size; };
    T operator []( int index);
    void pop_front();
    void pop_back();
    Node<T>* push_back(T data);
    Node<T>* push_front(T data);
    void clear();
    Node<T>* insert(T dane, int index);
    Node<T>* getAt(int index);
    void erase(int index);
};
template<typename T>
List<T>::List()
{
    Size = 0;
    head = nullptr;
    tail = nullptr;
}

template<typename T>
List<T>::~List()
{
    clear();
}

template<typename T>
Node<T>* List<T>::getAt(int index) {
    Node<T>* ptr = head;
    int n = 0;
    while (n != index) {
        if (ptr == nullptr) {
            return ptr;
        }
        ptr = ptr->next;
        n++;
    }
    return ptr;
}

template<typename T>
Node<T>* List<T>::push_back(T data)
{
    Node<T>* temp = new Node<T>(data);
    temp->prev = tail;
    if (tail != nullptr)
        tail->next = temp;
    if (head == nullptr)
        head = temp;
    tail = temp;
    Size++;
    return temp;
}

template<typename T>
T List<T> :: operator[](int index)
{
    int counter = 0;
    Node<T>*current = this->head;
    while (current != nullptr) {
        if (counter == index) {
            return current->data;
        }
        current = current->next;
        counter++;
    }
}

template<typename T>
void List<T>::pop_front()
{
    if (head == nullptr)
        return;
    Node<T>* temp = head->next;
    if (temp != nullptr)
        temp->prev = nullptr;
    else
        tail = nullptr;
    delete head;
    head = temp;
    Size--;
}

template<typename T>
Node<T>* List<T>::push_front(T data)
{
    Node<T>* temp = new Node<T>(data);
    temp->next = head;
    if (head != nullptr)
        head->prev = temp;
    if (tail == nullptr)
        tail = temp;
    head = temp;
    Size++;
    return temp;
}

template<typename T>
void List<T>::clear()
{
    while (Size) {
        pop_front();
    }
}

template<typename T>
void List<T>::pop_back()
{
    if (tail == nullptr)
        return;
    Node<T>* ptr = tail->prev;
    if (ptr != nullptr)
        ptr->next = nullptr;
    else
        head = nullptr;
    delete tail;
    tail = ptr;
    Size--;
}

template<typename T>
Node<T>* List<T>::insert(T dane, int index)
{
    Node<T>* right = getAt(index);
    if (right == nullptr)
        return push_back(data);

    Node<T>* left = right->prev;
    if (left == nullptr)
        return push_front(data);

    Node<T>* ptr = new Node<T>(data);
    ptr->prev = left;
    ptr->next = right;
    left->next = ptr;
    right->prev = ptr;
    Size++;
    return ptr;
}

template<typename T>
void List<T>::erase(int index)
{
    Node<T>* ptr = getAt(index);
    if (ptr == nullptr)
        return;
    if (ptr->pPrev = nullptr) {
        pop_front();
        return;
    }
    if (ptr->next == nullptr) {
        pop_back();
        return;
    }
    Node<T>* left = ptr->prev;
    Node<T>* right = ptr->next;
    left->next = right;
    right->prev = left;
    delete ptr;
    Size--;
}
struct some_object
{
    int field_1;
    char field_2;
};

int main()
{   
    List<int> ls1;
    clock_t t1 = clock();
    for (int i = 0; i < 1000; i++) {
        ls1.push_front(rand()%1000);
    }
    clock_t t2 = clock();
    double time = (t2 - t1) / (double)CLOCKS_PER_SEC;
   
    clock_t t3 = clock();
    for (int i = 0; i < 1000; i++) {
        cout << ls1[i]<<"\t";
    }
    clock_t t4 = clock();
    double time1 = (t4 - t3) / (double)CLOCKS_PER_SEC;
    cout << "\nStwozyl 1000 elementow za: " << time;
    cout << "\tPokazal na ekranie 1000 elementow za: " << time1;

    List < some_object >* ll = new List < some_object >();
    some_object so;
    so.field_1 = 123;
    so.field_2 = 'avs';
    ll->push_back(so);
    cout <<"\n" << &ll;
}