package swordFingerProvided;

import java.util.Arrays;

/**
 * @author Kezi
 * @date 2022年06月28日 0:06
 */
public class Solution {

	// 青蛙跳台阶问题
	public int numWays(int n) {
		int a = 1, b = 1, sum;
		for (int i = 0; i < n; i++) {
			sum = (a + b) % 1000000007;
			a = b;
			b = sum;
		}
		return a;
	}

	// 股票的最大利润
	public int maxProfit(int[] prices) {
		int cost = Integer.MAX_VALUE, profit = 0;
		for (int price : prices) {
			cost = Math.min(cost, price);
			profit = Math.max(profit, price - cost);
		}
		return profit;
	}

	// 连续子数组的最大和
	public int maxSubArray(int[] nums) {
		int res = nums[0];
		for (int i = 1; i < nums.length; i++) {
			nums[i] += Math.max(nums[i - 1], 0);
			res = Math.max(res, nums[i]);
		}
		return res;
	}

	/**
	 * 礼物的最大价值
	 * 根据题目说明，易得某单元格只可能从上边单元格或左边单元格到达
	 */
	public int maxValue(int[][] grid) {
		int m = grid.length, n = grid[0].length;
		// 初始化第一行、第一列,用于优化
		for (int j = 1; j < n; j++) {
			grid[0][j] += grid[0][j - 1];
		}
		for (int i = 1; i < m; i++) {
			grid[i][0] += grid[i - 1][0];
		}
		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				grid[i][j] = Math.max(grid[i][j - 1], grid[i - 1][j]);
			}
		}
		return grid[m - 1][n - 1];
	}

	/**
	 * 数组组成最小的数
	 * 若拼接字符串x+y>y+x ，则 x “大于” y ；
	 * 反之，若 x + y < y + x ，则 x “小于” y ；
	 */
	public String minNumber(int[] nums) {
		String[] strs = new String[nums.length];
		for (int i = 0; i < nums.length; i++) {
			strs[i] = String.valueOf(nums[i]);
		}
		Arrays.sort(strs, (x, y) -> (x + y).compareTo(y + x));
		StringBuilder res = new StringBuilder();
		for (String s : strs) {
			res.append(s);
		}
		return res.toString();
	}

	/**
	 * 求1+2+3+。。。+n
	 * 要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）
 	 */
	public int sumNums(int n) {
		boolean x = n > 1 && (n += sumNums((n - 1))) > 0;
		return n;
	}
}
