package prepareHW;

import java.util.*;

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
	public static void main(String[] args) {
		Scanner scanner = new Scanner(System.in);
		String[] strings = scanner.nextLine().split(" ");
		Set<String> set = new HashSet<>();
		for (String str : strings) {
			set.add(str);
		}
		String truePass = "";
		for (String str : strings) {
			boolean flag = true;
			for (int i = 1; i < str.length(); i++) {
				String substring = str.substring(0, i);
				if (!set.contains(substring)) {
					flag = false;
					break;
				}
			}
			if (flag) {
				if (str.length() > truePass.length()) {
					truePass = str;
				}
				if (str.length() == truePass.length() && str.compareTo(truePass) > 0) {
					truePass = str;
				}
			}
		}
		System.out.println(truePass);
	}

}
