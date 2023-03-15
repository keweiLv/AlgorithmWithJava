package prepareHW;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

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

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		String moneyStr = sc.nextLine();
		int money = Integer.parseInt(moneyStr);
		String prices = sc.nextLine();
		String[] split = prices.split(" ");
		List<Integer> list = new ArrayList<>();
		for (String str : split) {
			list.add(Integer.parseInt(str));
		}
		int[] dp = new int[money + 1];
		int n = list.size();
		for (int i = 1; i <= n; i++) {
			int num = list.get(i-1);
			for (int j = money;j >= i;j--){
				int pre = dp[j-i];
				if (j >= i && pre != Integer.MIN_VALUE){
					dp[j] = Math.max(dp[j],pre + num);
				}
			}
		}
		System.out.println(dp[money]);
		sc.close();
	}

}
