#pragma once
#include <iostream>
#include <limits>
#include <string>
#include <cmath>
#include <sstream>
using namespace std;

template <typename T>
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

template <typename T>
class List {
public:
    Node<T>* head;
    Node<T>* tail;
    int Size;
    List();
    ~List();

    int Get_size() { return Size; };
    T operator[](int index);
    void pop_front();
    void pop_back();
    Node<T>* push_back(T data);
    Node<T>* push_front(T data);
    void clear();
    Node<T>* insert(T data, int index);
    Node<T>* getAt(int index);
    void erase(Node<T>* node);
};

template <typename T>
List<T>::List() {
    Size = 0;
    head = nullptr;
    tail = nullptr;
}

template <typename T>
List<T>::~List() {
    clear();
}

template <typename T>
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

template <typename T>
Node<T>* List<T>::push_back(T data) {
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

template <typename T>
T List<T>::operator[](int index) {
    int counter = 0;
    Node<T>* current = this->head;
    while (current != nullptr) {
        if (counter == index) {
            return current->data;
        }
        current = current->next;
        counter++;
    }
}

template <typename T>
void List<T>::pop_front() {
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

template <typename T>
Node<T>* List<T>::push_front(T data) {
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

template <typename T>
void List<T>::clear() {
    while (Size) {
        pop_front();
    }
}

template <typename T>
void List<T>::pop_back() {
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

template <typename T>
Node<T>* List<T>::insert(T data, int index) {
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
// ...

template <typename T>
void List<T>::erase(Node<T>* node) {
    if (node == nullptr)
        return;

    if (node->prev == nullptr) {
        pop_front();
        return;
    }

    if (node->next == nullptr) {
        pop_back();
        return;
    }

    Node<T>* left = node->prev;
    Node<T>* right = node->next;
    left->next = right;
    right->prev = left;
    delete node;
    Size--;
}