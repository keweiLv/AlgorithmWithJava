package swordFingerProvided;

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
		int cost = Integer.MAX_VALUE,profit = 0;
		for (int price:prices){
			cost = Math.min(cost,price);
			profit = Math.max(profit,price - cost);
		}
		return profit;
	}

	// 连续子数组的最大和
	public int maxSubArray(int[] nums) {
		int res = nums[0];
		for (int i = 1; i < nums.length; i++) {
			nums[i] += Math.max(nums[i-1],0);
			res = Math.max(res,nums[i]);
		}
		return res;
	}
}
