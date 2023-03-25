package prepareHW;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Kezi
 * @date 2023年03月12日 21:15
 */
public class Main {

//	public static void main(String[] args) {
//		Scanner scanner = new Scanner(System.in);
//		int m = scanner.nextInt();
//		int n = scanner.nextInt();
//		scanner.nextLine();
//		List<LinkedList<Integer>> list = new ArrayList<>();
//		for (int i = 0; i < n; i++) {
//			LinkedList<Integer> collect = Arrays.stream(scanner.nextLine().split(",")).map(Integer::parseInt).collect(Collectors.toCollection(LinkedList::new));
//			list.add(collect);
//		}
//		StringBuilder sb = new StringBuilder();
//		int index = 0;
//		while (list.size() > 0) {
//			LinkedList<Integer> integers = list.get(index);
//			for (int i = 0; i < m; i++) {
//				if (integers.isEmpty()) {
//					list.remove(integers);
//					index--;
//					break;
//				}
//				sb.append(integers.removeFirst()).append(",");
//			}
//			index++;
//			if (index >= list.size()) {
//				index = 0;
//			}
//		}
//		System.out.println(sb.toString());
//	}

//	public static void main(String[] args) {
//		Scanner sc = new Scanner(System.in);
//		String moneyStr = sc.nextLine();
//		int money = Integer.parseInt(moneyStr);
//		String prices = sc.nextLine();
//		String[] split = prices.split(" ");
//		List<Integer> list = new ArrayList<>();
//		for (String str : split) {
//			list.add(Integer.parseInt(str));
//		}
//		int[] dp = new int[money + 1];
//		int n = list.size();
//		for (int i = 1; i <= n; i++) {
//			int num = list.get(i-1);
//			for (int j = money;j >= i;j--){
//				int pre = dp[j-i];
//				if (j >= i && pre != Integer.MIN_VALUE){
//					dp[j] = Math.max(dp[j],pre + num);
//				}
//			}
//		}
//		System.out.println(dp[money]);
//		sc.close();
//	}


    // 最差的产品序列
//	public static void main(String[] args) {
//		Scanner scanner = new Scanner(System.in);
//		int m = scanner.nextInt();
//		Integer[] array = Arrays.stream(scanner.next().split(",")).map(Integer::parseInt).toArray(Integer[]::new);
//		ArrayList<Integer> ans = new ArrayList<>();
//		for (int i = 0; i <= array.length - m; i++) {
//			int min = Integer.MAX_VALUE;
//			for (int j = i; j < i + m; j++) {
//				min = Math.min(min, array[j]);
//			}
//			ans.add(min);
//		}
//		StringJoiner sj = new StringJoiner(",");
//		for (Integer num:ans){
//			sj.add(num + "");
//		}
//		System.out.println(sj.toString());
//	}

    // 最大利润
//	public static void main(String[] args) {
//		Scanner scanner = new Scanner(System.in);
//		int items = scanner.nextInt();
//		int day = scanner.nextInt();
//		List<Integer> maxItems = new ArrayList<>();
//		for (int i = 0; i < items; i++) {
//			maxItems.add(scanner.nextInt());
//		}
//		List<List<Integer>> prices = new ArrayList<>();
//		for (int i = 0; i < items; i++) {
//			List<Integer> price = new ArrayList<>();
//			for (int j = 0; j < day; j++) {
//				price.add(scanner.nextInt());
//			}
//			prices.add(price);
//		}
//		int maxProfit = 0;
//		for (int i = 0; i < prices.size(); i++) {
//			int profit = 0;
//			for (int j = 1; j < prices.get(i).size(); j++) {
//				profit += Math.max(0,prices.get(i).get(j) - prices.get(i).get(j-1));
//			}
//			maxProfit += profit * maxItems.get(i);
//		}
//		System.out.println(maxProfit);
//	}

    // 字符串重新排序
//	public static void main(String[] args) {
//		Scanner scanner = new Scanner(System.in);
//		String[] s = scanner.nextLine().split(" ");
//		s = Arrays.stream(s).map(str -> {
//			char[] chars = str.toCharArray();
//			Arrays.sort(chars);
//			return new String(chars);
//		}).toArray(String[]::new);
//		HashMap<String, Integer> map = new HashMap<>();
//		for (String str : s) {
//			map.put(str, map.getOrDefault(str, 0) + 1);
//		}
//		Arrays.sort(s, (a, b) ->
//				!map.get(a).equals(map.get(b)) ? map.get(b) - map.get(a) : a.length() != b.length() ? a.length() - b.length() : a.compareTo(b));
//		StringJoiner sj = new StringJoiner(" ","","");
//		for (String s1:s){
//			sj.add(s1);
//		}
//		System.out.println(sj.toString());
//	}

    //最长的密码
//	public static void main(String[] args) {
//		Scanner scanner = new Scanner(System.in);
//		String[] strings = scanner.nextLine().split(" ");
//		Set<String> set = new HashSet<>();
//		for (String str : strings) {
//			set.add(str);
//		}
//		String truePass = "";
//		for (String str : strings) {
//			boolean flag = true;
//			for (int i = 1; i < str.length(); i++) {
//				String substring = str.substring(0, i);
//				if (!set.contains(substring)) {
//					flag = false;
//					break;
//				}
//			}
//			if (flag) {
//				if (str.length() > truePass.length()) {
//					truePass = str;
//				}
//				if (str.length() == truePass.length() && str.compareTo(truePass) > 0) {
//					truePass = str;
//				}
//			}
//		}
//		System.out.println(truePass);
//	}

    // 通信误码
//    public static void main(String[] args) {
//        Scanner scanner = new Scanner(System.in);
//        String getN = scanner.nextLine();
//        int n = Integer.parseInt(getN);
//        String getM = scanner.nextLine();
//        String[] s = getM.split(" ");
//        ArrayList<Integer> arrs = new ArrayList<>();
//        for (String str : s) {
//            arrs.add(Integer.parseInt(str));
//        }
//        int maxCnt = 0;
//        Map<Integer, Integer> record = new HashMap<>();
//        for (int i = 0; i < n; i++) {
//            record.put(arrs.get(i), record.getOrDefault(arrs.get(i), 0) + 1);
//            maxCnt = Math.max(maxCnt, record.get(arrs.get(i)));
//        }
//        Set<Integer> maxNums = new HashSet<>();
//        for (Map.Entry<Integer, Integer> entry : record.entrySet()) {
//            if (entry.getValue() == maxCnt) {
//                maxNums.add(entry.getKey());
//            }
//        }
//        int ans = n;
//        for (int num : maxNums) {
//            int left = 0, right = n - 1;
//            while (arrs.get(left) != num) {
//                left++;
//            }
//            while (arrs.get(right) != num) {
//                right--;
//            }
//            if (left <= right) {
//                ans = Math.min(ans, right - left + 1);
//            }
//        }
//        System.out.println(ans);
//    }

	//最长的密码
//	public static void main(String[] args) {
//		Scanner scanner = new Scanner(System.in);
//		String[] strings = scanner.nextLine().split(" ");
//		Set<String> set = new HashSet<>();
//		for (String str : strings) {
//			set.add(str);
//		}
//		String truePass = "";
//		for (String str : strings) {
//			boolean flag = true;
//			for (int i = 1; i < str.length(); i++) {
//				String substring = str.substring(0, i);
//				if (!set.contains(substring)) {
//					flag = false;
//					break;
//				}
//			}
//			if (flag) {
//				if (str.length() > truePass.length()) {
//					truePass = str;
//				}
//				if (str.length() == truePass.length() && str.compareTo(truePass) > 0) {
//					truePass = str;
//				}
//			}
//		}
//		System.out.println(truePass);
//	}

	// 开心消消乐
//	public static void main(String[] args) {
//		Scanner in = new Scanner(System.in);
//		int rows = in.nextInt();
//		int cols = in.nextInt();
//		int[][] mat = new int[rows][cols];
//		for (int i = 0; i < rows; i++) {
//			for (int j = 0; j < cols; j++) {
//				mat[i][j] = in.nextInt();
//			}
//		}
//		int ans = 0;
//		for (int i = 0; i < rows; i++) {
//			for (int j = 0; j < cols; j++) {
//				if (mat[i][j] == 1){
//					ans++;
//					dfs(mat,i,j);
//				}
//			}
//		}
//		System.out.println(ans);
//	}
//	private static void dfs(int[][] mat, int i, int j) {
//		mat[i][j] = 0;
//		int rows = mat.length;
//		int cols = mat[0].length;
//		int[][] direct = {{-1,0},{1,0},{0,-1},{-1,0},{-1,-1},{-1,1},{1,-1},{1,1}};
//		for (int[] dir : direct){
//			int nxetX = i + dir[0];
//			int nextY = j + dir[1];
//			if (nxetX >=0 && nxetX < rows && nextY >= 0 && nextY < cols && mat[nxetX][nextY] == 1){
//				dfs(mat,nxetX,nextY);
//			}
//		}
//	}

	// 最大化控制资源成本
//	public static void main(String[] args) {
//		Scanner sc = new Scanner(System.in);
//		int num = sc.nextInt();
//		int[][] tasks = new int[num][3];
//		for (int i = 0; i < num; i++) {
//			tasks[i][0] = sc.nextInt();
//			tasks[i][1] = sc.nextInt();
//			tasks[i][2] = sc.nextInt();
//		}
//		Arrays.sort(tasks, (a, b) -> a[0] - b[0]);
//		PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] - b[0]);
//		int max = 0;
//		int cur = 0;
//		for (int[] task : tasks) {
//			while (!pq.isEmpty() && pq.peek()[0] < task[0]) {
//				int[] pop = pq.poll();
//				cur -= pop[1];
//			}
//			pq.offer(new int[]{task[1], task[2]});
//			cur += task[2];
//			max = Math.max(max, cur);
//		}
//		System.out.println(max);
//	}

	// 预定酒店
	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in);
		int n = scanner.nextInt();
		int k = scanner.nextInt();
		int x = scanner.nextInt();
		int[] prices = new int[n];
		for (int i = 0; i < n; i++) {
			prices[i] = scanner.nextInt();
		}
		Arrays.sort(prices);
		int[][] dif = new int[n][2];
		for (int i = 0; i < n; i++) {
			int price = prices[i];
			dif[i][0] = price;
			dif[i][1] = Math.abs(price - x);
		}
		List<int[]> sorted = Arrays.stream(dif).sorted(Comparator.comparing(item -> item[1])).collect(Collectors.toList());
		List<Integer> pick = new ArrayList<>();
		for (int i = 0; i < k; i++) {
			pick.add(sorted.get(i)[0]);
		}
		pick.sort(Integer::compareTo);
		for (int i = 0;i<pick.size();i++){
			System.out.print(pick.get(i));
			if (i != pick.size() -1){
				System.out.print(" ");
			}
		}
	}
}
