package acwing;

import java.io.*;

/**
 * @author Kezi
 * @date 2023年01月03日 23:38
 */
public class Main {


	static int N = 100010;
	static int[] a = new int[N];
	static int[] b = new int[N];


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
//	public static void main(String[] args) throws IOException {
//		BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
//		BigInteger a = new BigInteger(reader.readLine());
//		BigInteger b = new BigInteger(reader.readLine());
//		System.out.println(a.add(b));
//		reader.close();
//	}

	// 前缀和
//	public static void main(String[] args) {
//		Scanner sc = new Scanner(System.in);
//		int n = sc.nextInt();
//		int m = sc.nextInt();
//		int[] arr = new int[N];
//		for (int i = 1; i <= n; i++) {
//			arr[i] = sc.nextInt();
//		}
//		int[] s = new int[N];
//		s[0] = 0;
//		for (int i = 1; i <= n; i++) {
//			s[i] = s[i - 1] + arr[i];
//		}
//		while (m-- > 0) {
//			int l = sc.nextInt();
//			int r = sc.nextInt();
//			System.out.println(s[r] - s[l-1]);
//		}
//	}

	// 二进制中1的个数
//	public static void main(String[] args) {
//		Scanner sc = new Scanner(System.in);
//		int n = sc.nextInt();
//		while (n-- != 0) {
//			int x = sc.nextInt();
//			int res = 0;
//			while (x != 0){
//				x &= (x-1);
//				res++;
//			}
//			System.out.print(res + " ");
//		}
//	}

//    // 最长连续不重复子序列
//    public static void main(String[] args) throws IOException {
//        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
//        int n = Integer.parseInt(reader.readLine());
//        int[] nums = new int[n];
//        String[] str = reader.readLine().split(" ");
//        for (int i = 0; i < n; i++) {
//            nums[i] = Integer.parseInt(str[i]);
//        }
//        Map<Integer, Integer> map = new HashMap<>();
//        int res = 0;
//        for (int i = 0, j = 0; i < n; i++) {
//            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
//            while (map.get(nums[i]) > 1) {
//                map.put(nums[j], map.get(nums[j]) - 1);
//                j++;
//            }
//            res = Math.max(res,i-j+1);
//        }
//        System.out.println(res);
//    }

	// 数组元素的目标和
//    public static void main(String[] args) throws IOException {
//        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
//        String[] strings = reader.readLine().split(" ");
//        int n = Integer.parseInt(strings[0]);
//        int m = Integer.parseInt(strings[1]);
//        int x = Integer.parseInt(strings[2]);
//
//        String[] A = reader.readLine().split(" ");
//        for (int i = 0; i < n; i++) {
//            a[i] = Integer.parseInt(A[i]);
//        }
//        String[] B = reader.readLine().split(" ");
//        for (int i = 0; i < m; i++) {
//            b[i] = Integer.parseInt(B[i]);
//        }
//        int i = 0, j = m - 1;
//        while (i < n && j >= 0) {
//            if (a[i] + b[j] > x){
//                j--;
//            }else if (a[i] + b[j] < x){
//                i++;
//            }else {
//                System.out.println(i + " " + j);
//                break;
//            }
//        }
//    }

	// 判断子序列
//	public static void main(String[] args) {
//		Scanner scanner = new Scanner(System.in);
//		int n = scanner.nextInt();
//		int m = scanner.nextInt();
//		int[] a = new int[n];
//		int[] b = new int[m];
//		for (int i = 0; i < n; i++) {
//			a[i] = scanner.nextInt();
//		}
//		for (int i = 0; i < m; i++) {
//			b[i] = scanner.nextInt();
//		}
//		int i = 0, j = 0;
//		while (i < n && j < m) {
//			if (a[i] == b[j]) {
//				i++;
//			}
//			j++;
//		}
//		if (i == n) {
//			System.out.println("Yes");
//		} else {
//			System.out.println("No");
//		}
//	}

	// 区间合并
//	public static void main(String[] args) {
//		List<Interval> intervals = new ArrayList<>();
//		read(intervals);
//		Collections.sort(intervals);
//
//		int start = intervals.get(0).start;
//		int end = intervals.get(0).end;
//		int total = 0;
//		for (Interval interval : intervals) {
//			if (interval.start <= end) {
//				end = Math.max(end, interval.end);
//			} else {
//				start = interval.start;
//				end = interval.end;
//				total++;
//			}
//		}
//		System.out.println(total + 1);
//	}
//	private static void read(List<Interval> intervals) {
//		Scanner scanner = new Scanner(System.in);
//		int n = scanner.nextInt();
//		while (n-- > 0) {
//			int start = scanner.nextInt();
//			int end = scanner.nextInt();
//			intervals.add(new Interval(start, end));
//		}
//		scanner.close();
//	}
//
//	static class Interval implements Comparable<Interval> {
//		public int start, end;
//
//		public Interval(int start, int end) {
//			this.start = start;
//			this.end = end;
//		}
//
//		@Override
//		public int compareTo(Interval object) {
//			return Integer.compare(start, object.start);
//		}
//	}

	// 单指针
//	public static int[] e = new int[N];
//	public static int[] ne = new int[N];
//	public static int head = -1;
//	public static int idx = 0;
//
//	public static void main(String[] args) {
//		Scanner scanner = new Scanner(System.in);
//		int n = scanner.nextInt();
//		while (n-- > 0) {
//			String str = scanner.next();
//			if (str.equals("H")) {
//				int x = scanner.nextInt();
//				insertHead(x);
//			} else if (str.equals("I")) {
//				int k = scanner.nextInt();
//				int x = scanner.nextInt();
//				insert(k - 1, x);
//			} else if (str.equals("D")) {
//				int k = scanner.nextInt();
//				if (k == 0) {
//					head = ne[head];
//				} else {
//					delete(k - 1);
//				}
//			}
//		}
//		int i = head;
//		while (i != -1) {
//			System.out.print(e[i] + " ");
//			i = ne[i];
//		}
//	}
//
//	private static void delete(int k) {
//		ne[k] = ne[ne[k]];
//	}
//
//	private static void insert(int k, int x) {
//		e[idx] = x;
//		ne[idx] = ne[k];
//		ne[k] = idx;
//		idx++;
//	}
//
//	private static void insertHead(int x) {
//		e[idx] = x;
//		ne[idx] = head;
//		head = idx;
//		idx++;
//	}

	// 模拟栈
	static int[] s = new int[N];
	static int tt = 0;

//	public static void main(String[] args) {
//		String op = null;
//		Scanner scanner = new Scanner(System.in);
//		int m = scanner.nextInt();
//		while (m-- > 0) {
//			op = scanner.next();
//			switch (op) {
//				case "push":
//					push(scanner.nextInt());
//					break;
//				case "pop":
//					pop();
//					break;
//				case "empty":
//					System.out.println(empty() ? "YES" : "NO");
//					break;
//				case "query": {
//					System.out.println(query());
//					break;
//				}
//				default:
//			}
//		}
//	}

	private static int query() {
		return s[tt];
	}

	private static boolean empty() {
		return tt == 0;
	}

	private static void pop() {
		tt--;
	}

	private static void push(int x) {
		s[++tt] = x;
	}

	// 滑动窗口
	public static void main(String[] args) throws IOException {
		BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
		PrintWriter wt = new PrintWriter(new OutputStreamWriter(System.out));
		String[] st = reader.readLine().split(" ");
		int n = Integer.parseInt(st[0]);
		int k = Integer.parseInt(st[1]);
		String[] str = reader.readLine().split(" ");
		for (int i = 0; i < n; i++) {
			a[i] = Integer.parseInt(str[i]);
		}
		int hh = 0, tt = -1;
		int[] q = new int[N];
		for (int i = 0; i < n; i++) {
			if (hh <= tt && q[hh] < i - k + 1) {
				hh++;
			}
		}
	}
}
