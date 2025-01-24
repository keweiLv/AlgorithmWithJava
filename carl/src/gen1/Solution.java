package carl.src.gen1;

/**
 * @author kezi
 */
public class Solution {
    // 二分查找
    public int search(int[] nums, int target) {
        if (target < nums[0] || target > nums[nums.length - 1]) {
            return -1;
        }
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) >> 1;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }

    // 移除元素
    public int removeElement(int[] nums, int val) {
        int slow = 0;
        for (int fast = 0; fast < nums.length; fast++) {
            if (nums[fast] != val) {
                nums[slow] = nums[fast];
                slow++;
            }
        }
        return slow;
    }

    // 购买水果需要的最少金币
    public int minimumCoins(int[] prices) {
        int n = prices.length;
        int[] memo = new int[(n + 1) / 2];
        return minimumCoinsDfs(1, prices, memo);
    }

    private int minimumCoinsDfs(int i, int[] prices, int[] memo) {
        if (i * 2 >= prices.length) {
            return prices[i - 1];
        }
        if (memo[i] != 0) {
            return memo[i];
        }
        int res = Integer.MAX_VALUE;
        for (int j = i + 1; j <= i * 2 + 1; j++) {
            res = Math.min(res, minimumCoinsDfs(j, prices, memo));
        }
        return memo[i] = res + prices[i - 1];
    }
}
