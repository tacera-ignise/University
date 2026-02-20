#include <iostream>
#include <algorithm>
using namespace std;
enum Color { RED, BLACK };

template <typename T>
struct Node {
    T data;
    Node* parent;
    Node* left;
    Node* right;
    Color color;
   
    Node(T val)
        : data(val), parent(nullptr), left(nullptr), right(nullptr), color(RED) {}
};

template <typename T>
class RBT{
private:
    Node<T>* root;

    void leftRotate(Node<T>* x) {
        Node<T>* y = x->right;
        x->right = y->left;
        if (y->left != nullptr) {
            y->left->parent = x;
        }
        y->parent = x->parent;
        if (x->parent == nullptr) {
            root = y;
        }
        else if (x == x->parent->left) {
            x->parent->left = y;
        }
        else {
            x->parent->right = y;
        }
        y->left = x;
        x->parent = y;
    }

    void rightRotate(Node<T>* y) {
        Node<T>* x = y->left;
        y->left = x->right;
        if (x->right != nullptr) {
            x->right->parent = y;
        }
        x->parent = y->parent;
        if (y->parent == nullptr) {
            root = x;
        }
        else if (y == y->parent->left) {
            y->parent->left = x;
        }
        else {
            y->parent->right = x;
        }
        x->right = y;
        y->parent = x;
    }

    void Conditions(Node<T>* z) {
        while (z != root && z->parent->color == RED) {
            if (z->parent == z->parent->parent->left) {
                Node<T>* y = z->parent->parent->right;
                if (y && y->color == RED) {
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                }
                else {
                    if (z == z->parent->right) {
                        z = z->parent;
                        leftRotate(z);
                    }
                    z->parent->color = BLACK;
                    z->parent->parent->color = RED;
                    rightRotate(z->parent->parent);
                }
            }
            else {
                Node<T>* y = z->parent->parent->left;
                if (y && y->color == RED) {
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                }
                else {
                    if (z == z->parent->left) {
                        z = z->parent;
                        rightRotate(z);
                    }
                    z->parent->color = BLACK;
                    z->parent->parent->color = RED;
                    leftRotate(z->parent->parent);
                }
            }
        }
        root->color = BLACK;
    }

    void Add(Node<T>* z) {
        Node<T>* y = nullptr;
        Node<T>* x = root;
        while (x != nullptr) {
            y = x;
            if (z->data < x->data) {
                x = x->left;
            }
            else {
                x = x->right;
            }
        }
        z->parent = y;
        if (y == nullptr) {
            root = z;
        }
        else if (z->data < y->data) {
            y->left = z;
        }
        else {
            y->right = z;
        }
        Conditions(z);
    }

    Node<T>* searchHelper(Node<T>* root, T key) {
        if (root == nullptr || root->data == key) {
            return root;
        }

        if (key < root->data) {
            return searchHelper(root->left, key);
        }

        return searchHelper(root->right, key);
    }

    Node<T>* minimum(Node<T>* x) {
        while (x->left != nullptr) {
            x = x->left;
        }
        return x;
    }

    void transplant(Node<T>* u, Node<T>* v) {
        if (u->parent == nullptr) {
            root = v;
        }
        else if (u == u->parent->left) {
            u->parent->left = v;
        }
        else {
            u->parent->right = v;
        }
        if (v != nullptr) {
            v->parent = u->parent;
        }
    }

    void deleteFixup(Node<T>* x) {
        while (x != nullptr && x != root && x->color == BLACK) {
            if (x == x->parent->left) {
                Node<T>* w = x->parent->right;
                if (w->color == RED) {
                    w->color = BLACK;
                    x->parent->color = RED;
                    leftRotate(x->parent);
                    w = x->parent->right;
                }
                if (w->left->color == BLACK && w->right->color == BLACK) {
                    w->color = RED;
                    x = x->parent;
                }
                else {
                    if (w->right->color == BLACK) {
                        w->left->color = BLACK;
                        w->color = RED;
                        rightRotate(w);
                        w = x->parent->right;
                    }
                    w->color = x->parent->color;
                    x->parent->color = BLACK;
                    w->right->color = BLACK;
                    leftRotate(x->parent);
                    x = root;
                }
            }
            else {
                Node<T>* w = x->parent->left;
                if (w->color == RED) {
                    w->color = BLACK;
                    x->parent->color = RED;
                    rightRotate(x->parent);
                    w = x->parent->left;
                }
                if (w->right->color == BLACK && w->left->color == BLACK) {
                    w->color = RED;
                    x = x->parent;
                }
                else {
                    if (w->left->color == BLACK) {
                        w->right->color = BLACK;
                        w->color = RED;
                        leftRotate(w);
                        w = x->parent->left;
                    }
                    w->color = x->parent->color;
                    x->parent->color = BLACK;
                    w->left->color = BLACK;
                    rightRotate(x->parent);
                    x = root;
                }
            }
        }
        if (x != nullptr) {
            x->color = BLACK;
        }
    }

    void remove(Node<T>* z) {
        Node<T>* y = z;
        Node<T>* x;
        Color yOriginalColor = y->color;

        if (z->left == nullptr) {
            x = z->right;
            transplant(z, z->right);
        }
        else if (z->right == nullptr) {
            x = z->left;
            transplant(z, z->left);
        }
        else {
            y = minimum(z->right);
            yOriginalColor = y->color;
            x = y->right;
            if (y->parent == z) {
                if (x != nullptr) {
                    x->parent = y;
                }
            }
            else {
                transplant(y, y->right);
                y->right = z->right;
                y->right->parent = y;
            }
            transplant(z, y);
            y->left = z->left;
            y->left->parent = y;
            y->color = z->color;
        }

        if (yOriginalColor == BLACK) {
            deleteFixup(x);
        }

        delete z;
    }

    int heightHelper(Node<T>* root) {
        if (root == nullptr) {
            return 0;
        }

        int leftHeight = heightHelper(root->left);
        int rightHeight = heightHelper(root->right);

        return max(leftHeight, rightHeight) + 1;
    }

    void clearHelper(Node<T>* root) {
        if (root != nullptr) {
            clearHelper(root->left);
            clearHelper(root->right);
            delete root;
        }
    }

public:
    RBT() : root(nullptr) {}

    ~RBT() {
        clear();
    }

    void Add(T val) {
        Node<T>* z = new Node<T>(val);
        Add(z);
    }

    Node<T>* search(T key) {
        return searchHelper(root, key);
    }

    void remove(T key) {
        Node<T>* z = search(key);
        if (z != nullptr) {
            remove(z);
        }
    }

    void clear() {
        clearHelper(root);
        root = nullptr;
    }

    int height() {
        return heightHelper(root);
    }

    void inorderHelper(Node<T>* root) {
        if (root != nullptr) {
            inorderHelper(root->left);
            cout << root->data << " ";
            inorderHelper(root->right);
        }
    }

    void inorder() {
        inorderHelper(root);
        cout << endl;
    }
};

int main() {
    RBT<int> tree;
    
    tree.Add('4');
    tree.Add('2');
    tree.Add('7');

    cout << "Inorder traversal: ";
    tree.inorder();

    cout << "Height of the tree: " << tree.height() << endl;

    tree.remove('2');

    cout << "Inorder traversal after deletion: ";
    tree.inorder();
  
    cout << "Height of the tree after deletion: " << tree.height() << endl<< tree.search('7')->data;

    tree.clear();

    return 0;
}
