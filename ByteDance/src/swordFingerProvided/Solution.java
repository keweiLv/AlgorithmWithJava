package swordFingerProvided;

import java.util.*;

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

	// 链表中倒数第K个节点
	public ListNode getKthFromEnd(ListNode head, int k) {
		ListNode fast = head, slow = head;
		for (int i = 0; i < k; i++) {
			fast = fast.next;
		}
		while (fast != null) {
			fast = fast.next;
			slow = slow.next;
		}
		return slow;
	}

	/**
	 * 把字符串转换为整数
	 * border=2147483647(Integer.MAX_VALUE)//10=214748364
	 */
	public int strToInt(String str) {
		int res = 0, border = Integer.MAX_VALUE / 10;
		int i = 0, sign = 1, length = str.length();
		if (length == 0) {
			return 0;
		}
		while (str.charAt(i) == ' ') {
			if (++i == length) {
				return 0;
			}
		}
		if (str.charAt(i) == '-') {
			sign = -1;
		}
		if (str.charAt(i) == '-' || str.charAt(i) == '+') {
			i++;
		}
		for (int j = i; j < length; j++) {
			if (str.charAt(j) < '0' || str.charAt(j) > '9') {
				break;
			}
			if (res > border || res == border && str.charAt(j) > '7') {
				return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
			}
			res = res * 10 + (str.charAt(j) - '0');
		}
		return sign * res;
	}

	// 二进制加法
	public String addBinary(String a, String b) {
		StringBuilder res = new StringBuilder();
		int i = a.length() - 1;
		int j = b.length() - 1;
		int carry = 0;
		while (i >= 0 || j >= 0) {
			int digitA = i >= 0 ? a.charAt(i--) - '0' : 0;
			int digitB = j >= 0 ? b.charAt(j--) - '0' : 0;
			int sum = digitA + digitB + carry;
			carry = sum >= 2 ? 1 : 0;
			sum = sum >= 2 ? sum - 2 : sum;
			res.append(sum);
		}
		if (carry == 1) {
			res.append(1);
		}
		return res.reverse().toString();
	}

	// 只出现一次的数字
	public int singleNumber(int[] nums) {
		Map<Integer, Integer> freq = new HashMap<Integer, Integer>();
		for (int num : nums) {
			freq.put(num, freq.getOrDefault(num, 0) + 1);
		}
		int ans = 0;
		for (Map.Entry<Integer, Integer> entry : freq.entrySet()) {
			int num = entry.getKey(), occ = entry.getValue();
			if (occ == 1) {
				ans = num;
				break;
			}
		}
		return ans;
	}

	/**
	 * 单词长度的最大乘积
	 * 位运算 + 预计算
	 * 时间复杂度：O((m + n)* n)
	 * 空间复杂度：O(n)
	 */
	public int maxProduct(String[] words) {
		Map<Integer, Integer> map = new HashMap<>();
		int n = words.length;
		for (int i = 0; i < n; i++) {
			int bitMask = 0;
			for (char c : words[i].toCharArray()) {
				bitMask |= (1 << c - 'a');
			}
			map.put(bitMask, Math.max(map.getOrDefault(bitMask, 0), words[i].length()));
		}
		int ans = 0;
		for (int x : map.keySet()) {
			for (int y : map.keySet()) {
				if ((x & y) == 0) {
					ans = Math.max(ans, map.get(x) * map.get(y));
				}
			}
		}
		return ans;
	}

	/**
	 * 左右两边子数组的和相等
	 * 总和为total,左侧元素之和为sum，则 2 * sum + nums[i] = total
	 */
	public int pivotIndex(int[] nums) {
		int total = Arrays.stream(nums).sum();
		int sum = 0;
		for (int i = 0; i < nums.length; i++) {
			if (2 * sum + nums[i] == total) {
				return i;
			}
			sum += nums[i];
		}
		return -1;
	}

	// 和为k的子数组
	public int subarraySum(int[] nums, int k) {
		int pre_sum = 0;
		int res = 0;
		HashMap<Integer,Integer> map = new HashMap<>(16);
		map.put(0, 1);
		for (int i : nums) {
			pre_sum += i;
			res += map.getOrDefault(pre_sum - k, 0);
			map.put(pre_sum, map.getOrDefault(pre_sum, 0) + 1);
		}
		return res;
	}

	// 翻转链表
	public ListNode reverseList(ListNode head) {
		ListNode prev = null;
		ListNode cur = head;
		while (cur != null){
			ListNode next = cur.next;
			cur.next = prev;
			prev = cur;
			cur = next;
		}
		return prev;
	}

	// 链表中的两数相加
	public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
		Deque<Integer> stack1 = new ArrayDeque<>();
		Deque<Integer> stack2 = new ArrayDeque<>();
		while (l1 != null) {
			stack1.push(l1.val);
			l1 = l1.next;
		}
		while (l2 != null) {
			stack2.push(l2.val);
			l2 = l2.next;
		}
		int carry = 0;
		ListNode ans = null;
		while (!stack1.isEmpty() || !stack2.isEmpty() || carry != 0) {
			int a = stack1.isEmpty() ? 0 : stack1.pop();
			int b = stack2.isEmpty() ? 0 : stack2.pop();
			int cur = a + b + carry;
			carry = cur / 10;
			cur %= 10;
			ListNode curNode = new ListNode(cur);
			curNode.next = ans;
			ans = curNode;
		}
		return ans;
	}

	// 删除链表的倒数第n个节点
	public ListNode removeNthFromEnd(ListNode head, int n) {
		ListNode slow = head;
		ListNode fast = head;
		for (int i = 0;i < n;i++){
			fast = fast.next;
		}
		if (fast == null){
			return head;
		}
		while (fast.next != null){
			slow = slow.next;
			fast = fast.next;
		}
		slow.next = slow.next.next;
		return head;
	}
}
