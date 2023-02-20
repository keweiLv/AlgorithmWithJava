package prepareForByte;

import java.util.*;

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

	// 字符串的排列
	List<String> res = new LinkedList<>();
	char[] c;

	public String[] permutation(String s) {
		c = s.toCharArray();
		dfs(0);
		return res.toArray(new String[res.size()]);
	}

	private void dfs(int x) {
		if (x == c.length - 1) {
			res.add(String.valueOf(c));
			return;
		}
		Set<Character> set = new HashSet<>();
		for (int i = x; i < c.length; i++) {
			if (set.contains(c[i])) {
				continue;
			}
			set.add(c[i]);
			swap(i, x);
			dfs(x + 1);
			swap(i, x);
		}
	}

	private void swap(int i, int x) {
		char tmp = c[i];
		c[i] = c[x];
		c[x] = tmp;
	}

	// 环形链表
	public ListNode detectCycle(ListNode head) {
		ListNode fast = head, slow = head;
		while (true) {
			if (fast == null || fast.next == null) {
				return null;
			}
			fast = fast.next.next;
			slow = slow.next;
			if (fast == slow) {
				break;
			}
		}
		fast = head;
		while (fast != slow) {
			fast = fast.next;
			slow = slow.next;
		}
		return fast;
	}
}
