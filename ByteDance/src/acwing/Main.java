package acwing;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigInteger;

/**
 * @author Kezi
 * @date 2023年01月03日 23:38
 */
public class Main {
//	public static void main(String[] args) {
//		Scanner sc = new Scanner(System.in);
//		int n = sc.nextInt();
//		int[] q = new int[n];
//		for (int i = 0; i < n; i++) {
//			q[i] = sc.nextInt();
//		}
//		quickSort(q, 0, n - 1);
//		for (int i = 0; i < n; i++) {
//			System.out.print(q[i] + " ");
//		}
//	}


	private static void quickSort(int[] q, int l, int r) {
		if (l >= r) {
			return;
		}
		int x = q[l + r >> 1], i = l - 1, j = r + 1;
		while (i < j) {
			while (q[++i] < x) ;
			while (q[--j] > x) ;
			if (i < j) {
				int t = q[i];
				q[i] = q[j];
				q[j] = t;
			}
		}
		quickSort(q, l, j);
		quickSort(q, j + 1, r);
	}


	// 第K个数
	static int N = 100010;
	static int[] A = new int[N];
	static int n, k;
//	public static void main(String[] args) {
//		Scanner sc = new Scanner(System.in);
//		n = sc.nextInt();
//		k = sc.nextInt();
//		for (int i = 0; i < n; i++) {
//			A[i] = sc.nextInt();
//		}
//		System.out.print(quickSortForK(0,n-1,k-1));
//	}
	public static int quickSortForK(int l, int r, int k) {
		if (l >= r) {
			return A[k];
		}
		int x = A[l], i = l - 1, j = r + 1;
		while (i < j) {
			do i++; while (A[i] < x);
			do j--; while (A[j] > x);
			if (i < j) {
				int temp = A[i];
				A[i] = A[j];
				A[j] = temp;
			}
		}
		if (k <= j) {
			return quickSortForK(l, j, k);
		} else {
			return quickSortForK(j + 1, r, k);
		}
	}

	// 高精度加法
	public static void main(String[] args) throws IOException {
		BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
		BigInteger a = new BigInteger(reader.readLine());
		BigInteger b = new BigInteger(reader.readLine());
		System.out.println(a.add(b));
		reader.close();
	}
}
