package prepareForByte;

import java.util.Arrays;

/**
 * @author Kezi
 * @date 2023年02月18日 0:10
 */
public class Solution {


	// 买卖股票的最佳时机
	public int maxProfit(int[] prices) {
		int minPrices = Integer.MAX_VALUE;
		int maxProfit = 0;
		for (int i = 0; i < prices.length; i++) {
			if (prices[i] < minPrices) {
				minPrices = prices[i];
			} else if (prices[i] - minPrices > maxProfit) {
				maxProfit = prices[i] - minPrices;
			}
		}
		return maxProfit;
	}

	// 打家劫舍Ⅱ
	public int rob(int[] nums) {
		if (nums.length == 0) {
			return 0;
		}
		if (nums.length == 1) {
			return nums[0];
		}
		return Math.max(myRob(Arrays.copyOfRange(nums, 0, nums.length - 1)), myRob(Arrays.copyOfRange(nums, 1, nums.length)));
	}

	private int myRob(int[] nums) {
		int pre = 0, cur = 0, tmp;
		for (int num : nums) {
			tmp = cur;
			cur = Math.max(pre + num, cur);
			pre = tmp;
		}
		return cur;
	}
}
