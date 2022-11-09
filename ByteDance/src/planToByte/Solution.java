package planToByte;

import java.util.*;

/**
 * @author Kezi
 * @date 2022年11月07日 22:54
 * @describe 2022 第 4 季度字节跳动面试真题
 */
public class Solution {
	// 翻转链表
	public ListNode reverseList(ListNode head) {
		ListNode pre = null;
		ListNode cur = head;
		while (cur != null) {
			ListNode next = cur.next;
			cur.next = pre;
			pre = next;
			cur = next;
		}
		return pre;
	}

	// 买卖股票的最佳时机
	public int maxProfit(int prices[]) {
		int min = Integer.MAX_VALUE;
		int max = 0;
		for (int i = 0; i < prices.length; i++) {
			if (prices[i] < min) {
				min = prices[i];
			} else if (prices[i] - min > max) {
				max = prices[i] - min;
			}
		}
		return max;
	}

	/**
	 * 无重复字符的最长子串
	 */
	public int lengthOfLongestSubstring(String s) {
		Map<Character, Integer> map = new HashMap<>();
		int max = 0;
		int left = 0;
		for (int i = 0; i < s.length(); i++) {
			if (map.containsKey(s.charAt(i))) {
				left = Math.max(left, map.get(s.charAt(i)) + 1);
			}
			map.put(s.charAt(i), i);
			max = Math.max(max, i - left + 1);
		}
		return max;
	}

	// 数组中第K大的元素
	public int findKthLargest(int[] nums, int k) {
		PriorityQueue<Integer> queue = new PriorityQueue<>(k, Comparator.comparing(a -> a));
		for (int i = 0; i < k; i++) {
			queue.offer(nums[i]);
		}
		for (int i = k; i < nums.length; i++) {
			if (nums[i] > queue.peek()) {
				queue.poll();
				queue.offer(nums[i]);
			}
		}
		return queue.peek();
	}

	// 三数之和
	public List<List<Integer>> threeSum(int[] nums) {
		List<List<Integer>> ans = new ArrayList<>();
		Arrays.sort(nums);
		int len = nums.length;
		for (int i = 0; i < len; i++) {
			if (nums[i] > 0	){
				return ans;
			}
			if (i > 0 && nums[i] == nums[i-1]){
				continue;
			}
			int cur = nums[i];
			int l = i+1,r = len-1;
			while (l < r){
				int tmp = cur + nums[l] + nums[r];
				if (tmp == 0){
					List<Integer> list = new ArrayList<>();
					list.add(cur);
					list.add(nums[l]);
					list.add(nums[r]);
					ans.add(list);
					while (l < r && nums[l+1] == nums[l]){
						++l;
					}
					while (l < r && nums[r-1] == nums[r]){
						--r;
					}
					++l;
					--r;
				}else if (tmp < 0){
					++l;
				}else {
					--r;
				}
			}
		}
		return ans;
	}

}
