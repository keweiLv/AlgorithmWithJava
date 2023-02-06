package planToByte;

import java.util.*;

/**
 * @author Kezi
 * @date 2023年02月03日 23:07
 */
public class NewSolution {

	// 二叉树着色游戏
	private int x, lsz, rsz;

	public boolean btreeGameWinningMove(TreeNode root, int n, int x) {
		this.x = x;
		dfs(root);
		return Math.max(Math.max(lsz, rsz), n - 1 - lsz - rsz) * 2 > n;
	}

	private int dfs(TreeNode node) {
		if (node == null) {
			return 0;
		}
		int ls = dfs(node.left);
		int rs = dfs(node.right);
		if (node.val == x) {
			lsz = ls;
			rsz = rs;
		}
		return ls + rs + 1;
	}

	// 你能构造出连续值的最大数目
	public int getMaximumConsecutive(int[] coins) {
		int m = 0;
		Arrays.sort(coins);
		for (int coin : coins) {
			if (coin > m + 1) {
				break;
			}
			m += coin;
		}
		return m + 1;
	}

	// 最长连续序列
	public int longestConsecutive(int[] nums) {
		Set<Integer> set = new HashSet<>();
		for (int num : nums) {
			set.add(num);
		}
		int longestLen = 0;
		for (int num : set) {
			if (!set.contains(num - 1)) {
				int curNum = num + 1;
				int curLen = 1;
				while (set.contains(curNum + 1)) {
					curLen++;
					curNum++;
				}
				longestLen = Math.max(longestLen, curLen);
			}
		}
		return longestLen;
	}

	// 有效的括号
	private static final Map<Character, Character> map = new HashMap<Character, Character>() {{
		put('{', '}');
		put('[', ']');
		put('(', ')');
		put('?', '?');
	}};

	public boolean isValid(String s) {
		if (s.length() > 0 && !map.containsKey(s.charAt(0))) {
			return false;
		}
		LinkedList<Character> stack = new LinkedList<Character>() {{
			add('?');
		}};
		for (Character c : s.toCharArray()) {
			if (map.containsKey(c)) {
				stack.addLast(c);
			} else if (!map.get(stack.removeLast()).equals(c)) {
				return false;
			}
		}
		return stack.size() == 1;
	}

	// 爬楼梯
	public int climbStairs(int n) {
		if (n <= 2) {
			return n;
		}
		int[] f = new int[n + 1];
		f[1] = 1;
		f[2] = 2;
		for (int i = 3; i <= n; i++) {
			f[i] = f[i - 1] + f[i - 2];
		}
		return f[n];
	}

	// 计算布尔二叉树的值
	public boolean evaluateTree(TreeNode root) {
		if (root.left == null) {
			return root.val == 1;
		}
		boolean l = evaluateTree(root.left);
		boolean r = evaluateTree(root.right);
		return root.val == 2 ? l || r : l && r;
	}

}