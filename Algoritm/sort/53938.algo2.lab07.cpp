//ALGO2 LS1 210A LAB07
//Daniil Protsvitaiev
//pd53938@zut.edu.pl
#include "kopiec.h"
#include"LISTA.h"
int my_rand(const int rand_num)
{
	int x = rand();
	x <<= 15;
	x ^= rand();
	x %= rand_num + 1;
	return x;
}

template <typename T>
int cmp_ints(const T a, const T b)
{
	return a - b;
}

template <typename T>
int cmp_ints_ptr(const T a, const T b)
{
	return *a - *b;
}

void counting_sort(int* array, const int n, const int m)
{
	int* counters = new int[m + 1];

	for (int i = 0; i < m + 1; i++)
		counters[i] = 0;

	for (int i = 0; i < n; i++)
		counters[array[i]]++;

	int index = 0;
	for (int i = 0; i < m + 1; i++)
		while (counters[i] > 0)
		{
			array[index] = i;
			counters[i]--;
			index++;
		}

	delete[] counters;
}

void bucket_sort(int* array, const int n, const int m)
{
	List<int>** buckets = new List<int>*[n];

	for (int i = 0; i < n; i++)
		buckets[i] = new List<int>();

	for (int i = 0; i < n; i++)
	{
		const int index = static_cast<size_t>(array[i]) * n / (m + 1);
		buckets[index]->sort_add(array[i], cmp_ints);
	}

	int index = 0;
	for (int i = 0; i < n; i++)
	{
		const Node<int>* current = buckets[i]->head;
		while (current)
		{
			array[index++] = current->data;
			current = current->next;
		}
		// delete buckets[i];
	}
	delete[] buckets;
}

template <typename T>
void bucket_sort(T* array, const int n, const double m, double (*key)(T), int (*cmp)(T, T))
{
	List<T>** buckets = new List<T>*[n];

	for (int i = 0; i < n; i++)
		buckets[i] = new List<T>();
	double w = m / n;
	for (int i = 0; i < n; i++)
	{
		const int index = static_cast<int>(key(array[i]) / w);
		buckets[index]->sort_add(array[i], cmp);
	}

	int index = 0;
	for (int i = 0; i < n; i++)
	{
		const Node<T>* current = buckets[i]->Head;
		while (current)
		{
			array[index++] = current->data;
			current = current->next;
		}
	}
	delete[] buckets;
}

int main()
{
	constexpr int MAX_ORDER = 3;
	const int m = static_cast<int>(pow(10, 3));

		const int n = static_cast<int>(pow(10, 3));
		int* array1 = new int[n];

		clock_t t1 = clock();
		for (int i = 0; i < n; i++)
		{
			const int rand_val = my_rand(m);
			array1[i] = rand_val;
			//cout << array1[i] << "\t";
		}
		clock_t t2 = clock();
		cout << t2 - t1;
		int* array2 = new int[n];
		int* array3 = new int[n];
		memcpy(array2, array1, n * sizeof(int));
		memcpy(array3, array1, n * sizeof(int));

		// sortowanie przez zliczanie ( do wykonania w miejscu )
		/*t1 = clock();
		counting_sort(array1, n, m);
		t2 = clock();
		cout << t2 - t1;
		cout << "-----------------------------------------------------\n\n";
		for (int i = 0; i < n; i++) {
			cout << array1[i] << "\t";
		}*/
	

		// sortowanie przez kopcowanie ( do wykonania w miejscu )
		int arraySize = sizeof(array2);
		
		Copiec<int> mas(array2, n);
		cout << "Heap after injecting array: " << mas.toString() << endl;
		t1 = clock();
		mas.sort();
		t2 = clock();
		cout << "\n\n\n\n\n\n";
		cout << "Heap before sorting: " << mas.toString() << endl;

		/*for (int i = 0; i < n; i++) {
			cout << array2[i] << "\t";
		}*/
		cout << "\n\n\n";

		// sortowanie kubelkowe ( do wykonania w miejscu )
		/*t1 = clock();
		for (int i = 0; i < n; i++) {
			cout << array3[i] << "\t";
		}
		bucket_sort(array3, n, m);
		for (int i = 0; i < n; i++) {
			cout << array3[i] << "\t";
		}
		t2 = clock();
		cout << t2 - t1;*/
		delete[] array1;
		delete[] array2;
		delete[] array3;
		return 0;
}

