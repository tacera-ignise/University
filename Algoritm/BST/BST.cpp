
#include <Windows.h>

#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include<stdlib.h>
#include "Table_dynamic.h"

using namespace std;

template <typename T>
class Node {
public:
    T data{};
    size_t key{};
    Node* left{};
    Node* right{};
    Node* parent{};

    explicit Node(T data)
        : data(data)
    {
    }
    ~Node()
    {
        if (data != nullptr) {
            delete data;
            data = nullptr;
        }
    }
};

template <typename T>
class BST {
    Node<T>* root_{};
    size_t size_{};

    Node<T>* insert(T data, Node<T>*& root, int (*cmp)(T, T), Node<T>* parent_node = nullptr)
    {
        if (!root)
            return new_node(data, parent_node);

        if (cmp(data, root->data) < 0)
            root->left = insert(data, root->left, cmp, root);
        else
            root->right = insert(data, root->right, cmp, root);

        return root;
    }

    Node<T>* find(T data, Node<T>* root, int (*cmp)(T, T))
    {
        if (root == nullptr || cmp(data, root->data) == 0)
            return root;

        if (cmp(data, root->data) > 0)
            return find(data, root->right, cmp);

        return find(data, root->left, cmp);
    }

    Node<T>* new_node(T data, Node<T>*& parent_node)
    {
        auto* node = new Node<T>{ data };
        node->left = node->right = nullptr;
        node->parent = parent_node;

        size_++;
        node->key = size_;

        return node;
    }

    void pre_order_traversal(Node<T>* root, Table<Node<T>*>& data)
    {
        if (!root)
            return;

        data.add(root);
        pre_order_traversal(root->left, data);
        pre_order_traversal(root->right, data);
    }

    void in_order_traversal(Node<T>* root, Table<Node<T>*>& data)
    {
        if (!root)
            return;

        in_order_traversal(root->left, data);
        data.add(root);
        in_order_traversal(root->right, data);
    }

    void delete_node(Node<T>*& root)
    {
        if (!root)
            return;

        delete_node(root->left);
        delete_node(root->right);

        size_--;
        delete root;
        root = nullptr;
    }

    size_t get_depth(Node<T>* root)
    {
        if (!root)
            return 0;

        size_t left_depth = get_depth(root->left);
        size_t right_depth = get_depth(root->right);

        size_t depth = static_cast<size_t>(1) + max(left_depth, right_depth);

        return depth;
    }

public:
    ~BST() { clear(); }
    void clear() { delete_node(root_); }

    void insert(T data, int (*cmp)(T, T))
    {
        Node<T>* node = insert(data, root_, cmp);
        if (root_ == nullptr)
            root_ = node;
    }

    void delete_node_by_ptr(Node<T>* node_to_delete)
    {
        if (!node_to_delete)
            return;

        Node<T>* parent = node_to_delete->parent;
        Node<T>* left_child = node_to_delete->left;
        Node<T>* right_child = node_to_delete->right;

        if (!left_child && !right_child) {
            if (parent) {
                if (parent->left == node_to_delete)
                    parent->left = nullptr;
                else
                    parent->right = nullptr;
            }
            delete node_to_delete;
            size_--;
        }
        else if (!left_child || !right_child) {
            Node<T>* child = left_child != nullptr ? left_child : right_child;
            if (parent) {
                if (parent->left == node_to_delete)
                    parent->left = child;
                else
                    parent->right = child;
            }
            child->parent = parent;
            delete node_to_delete;
            size_--;
        }
        else {
            Node<T>* successor = right_child;
            while (successor->left)
                successor = successor->left;

            *node_to_delete->data = *successor->data;
            node_to_delete->key = successor->key;
            delete_node_by_ptr(successor);
        }
    }

    void  in_order_traversal(Table<Node<T>*>& data)
    {
        in_order_traversal(root_, data);
    }

    void  pre_order_traversal(Table<Node<T>*>& data)
    {
        pre_order_traversal(root_, data);
    }

    string to_string(string(*get_node_fields)(T),
        const size_t nodes_amount = 0)
    {
        ostringstream array_string;
        const size_t depth = get_depth();
        array_string
            << "┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓\n"
            << left << setw(25) << "┃Tree size"
            << "┃" << setw(22) << size_ << "┃\n"
            << setw(25) << "┃Tree depth"
            << "┃" << setw(22) << depth << "┃\n"
            << setw(25) << "┃Tree reference"
            << "┃" << setw(22) << this << "┃\n"
            << setw(25) << "┃Tree log"
            << "┃" << setw(22) << depth / log2(size_) << "┃\n"
            << "┗━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━┛\n";

        if (size_ == 0)
            return array_string.str();

        const size_t max_index = nodes_amount == 0 ? min(size_, 10) : min(size_, nodes_amount);

        auto* pre_order_array = new Table<Node<T>*>();
        pre_order_traversal(*pre_order_array);

        array_string << "{\n";
        for (size_t i = 0; i < max_index; i++) {
            Node<T>* node = pre_order_array->operator[](i);

            string parent_node = node->parent == nullptr ? "NULL" : std::to_string(node->parent->key);
            string left_node = node->left == nullptr ? "NULL" : std::to_string(node->left->key);
            string right_node = node->right == nullptr ? "NULL" : std::to_string(node->right->key);

            array_string << "\t(" << node->key << ": "
                << "[p: " << parent_node << ", l: " << left_node
                << ", r: " << right_node
                << "], data: " << get_node_fields(node->data) << "),\n";
        }
        array_string << "}\n";

        for (size_t i = 0; i < pre_order_array->getSize(); i++)
            pre_order_array->operator[](i) = nullptr;

        delete pre_order_array;

        return array_string.str();
    }

    Node<T>* find(T data, int (*cmp)(T, T)) { return find(data, root_, cmp); }

    Node<T>* get_root() { return root_; }

    size_t get_size() const { return size_; }

    size_t get_depth() { return get_depth(root_); }
};

template <typename T>
int digital_cmp(T comparand, T comparator)
{
    if (*comparand > *comparator)
        return 1;
    if (*comparand < *comparator)
        return -1;
    return 0;
}

size_t my_rand(const size_t rand_num)
{
    size_t x = rand();
    x <<= 15;
    x ^= rand();
    x %= (rand_num + 1);
    return x;
}

template <typename T>
string get_num_data(T number)
{
    return to_string(*number);
}

int main()
{
    const int MAX_ORDER = 7;
    auto* bst = new BST<int*>();
    
    for (int o = 1; o <= MAX_ORDER; o++) {
        const int n = pow(10, o);

        clock_t t1 = clock();
        for (int i = 0; i < n; i++) {
            int* num = new int(my_rand(n * 2));
            bst->insert(num, digital_cmp);
        }
        clock_t t2 = clock();

        const double exec_time = (t2 - t1) / static_cast<double>(CLOCKS_PER_SEC) * 1000;

        cout << bst->to_string(get_num_data) << "Exec. Time: " << exec_time << " ms"
            << endl;

        const int m = pow(10, 3);
        int hits = 0;

        t1 = clock();
        for (int i = 0; i < m; i++) {
            int* num = new int(my_rand(n * 2));
            const Node<int*>* result = bst->find(num, digital_cmp);

            if (result)
                hits++;
            delete num;
        }
        t2 = clock();

        const double finding_time = (t2 - t1) / static_cast<double>(CLOCKS_PER_SEC) * 1000;

        cout << "Hits: " << hits << "\nFinding Time: " << finding_time << " ms\n"
            << endl;

        bst->clear();
    }
    
    delete bst;
    return 0;
}