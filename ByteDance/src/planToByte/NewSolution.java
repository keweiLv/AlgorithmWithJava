package planToByte;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

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
}