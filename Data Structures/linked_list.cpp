#include <iostream>

class Node
{
public:
    int data;
    Node* next;
    Node(int d) : data(d), next(nullptr) {}
};

class Stack
{
public:
    Node* top;
    Node* bottom;
    int size;
    Stack() : top(nullptr), bottom(nullptr), size(0) {}

bool is_empty()
    {
        return bottom == nullptr;
    }
void push(int n)
    {
        Node* new_node = new Node(n);
        if (is_empty())
        {
            top = new_node;
            bottom = new_node;
        }
        else
        {
            top->next = new_node;
            top = new_node;
        }
        size++;
    }
int pop()
{
    if (is_empty())
    {
        std::cout << "Stack is empty" << std::endl;
        return -1;
    }
    else if (top == bottom)
    {
        int num = top->data;
        bottom = nullptr;
        top = nullptr;
        return num;
    }
    else
    {
        Node* current = bottom;
        while (current->next!=top)
        {
            current = current->next;
        }
        int value = top->data;
        delete top;
        top = current;
        top->next = nullptr;
        size--;
        return value;
    }
}
int peek()
{
    if (is_empty())
    {
        std::cout << "Stack is empty" << std::endl;
        return -1;
    }
    else
    {
        return top->data;
    }
}

class Queue
{
public:
    Node* top;
    Node* bottom;
    int size;
    Queue() : top(nullptr), bottom(nullptr), size(0) {}

bool is_empty()
{
    return bottom == nullptr;
}
void enqueue(int n)
{
    
}
}
}

