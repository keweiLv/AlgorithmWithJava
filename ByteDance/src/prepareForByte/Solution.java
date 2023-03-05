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

	// 排序数组中只出现一次的数字
	public int singleNonDuplicate(int[] nums) {
		int n = nums.length, l = 0, r = nums.length - 1;
		int ans = -1;
		while (l <= r) {
			int mid = l + (r - l) / 2;
			if (mid < n - 1 && nums[mid] == nums[mid + 1]) {
				if (mid % 2 == 0) {
					l = mid + 2;
				} else {
					r = mid - 1;
				}
			} else if (mid > 0 && nums[mid] == nums[mid - 1]) {
				if (mid % 2 == 0) {
					r = mid - 2;
				} else {
					l = mid + 1;
				}
			} else {
				ans = nums[mid];
				break;
			}
		}
		return ans;
	}

	// 子集
	public List<List<Integer>> subsets(int[] nums) {
		List<List<Integer>> res = new ArrayList<>();
		backtrack(0, nums, res, new ArrayList<>());
		return res;
	}

	private void backtrack(int i, int[] nums, List<List<Integer>> res, ArrayList<Integer> tmp) {
		res.add(new ArrayList<>(tmp));
		for (int j = i; j < nums.length; j++) {
			tmp.add(nums[j]);
			backtrack(j + 1, nums, res, tmp);
			tmp.remove(tmp.size() - 1);
		}
	}

	// 重排链表
	public void reorderList(ListNode head) {
		if (head == null) {
			return;
		}
		ListNode mid = middleNode(head);
		ListNode l1 = head;
		ListNode l2 = mid.next;
		mid.next = null;
		l2 = reverseList(l2);
		mergeList(l1, l2);
	}

	private void mergeList(ListNode l1, ListNode l2) {
		ListNode l1_tmp;
		ListNode l2_tmp;
		while (l1 != null && l2 != null) {
			l1_tmp = l1.next;
			l2_tmp = l2.next;
			l1.next = l2;
			l1 = l1_tmp;
			l2.next = l1;
			l2 = l2_tmp;
		}
	}

	private ListNode reverseList(ListNode head) {
		ListNode prev = null;
		ListNode curr = head;
		while (curr != null) {
			ListNode next = curr.next;
			curr.next = prev;
			prev = curr;
			curr = next;
		}
		return prev;
	}

	private ListNode middleNode(ListNode head) {
		ListNode slow = head;
		ListNode fast = head;
		while (fast.next != null && fast.next.next != null) {
			slow = slow.next;
			fast = fast.next.next;
		}
		return slow;
	}

	// 下一个最大元素二
	public int[] nextGreaterElements(int[] nums) {
		int n = nums.length;
		int[] ans = new int[n];
		Arrays.fill(ans, -1);
		Deque<Integer> deque = new ArrayDeque<>();
		for (int i = 0; i < n * 2; i++) {
			while (!deque.isEmpty() && nums[i % n] > nums[deque.peekLast()]) {
				int poll = deque.pollLast();
				ans[poll] = nums[i % n];
			}
			deque.addLast(i % n);
		}
		return ans;
	}

	// 链表中倒数第K哥节点
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

	// 划分字母区间
	public List<Integer> partitionLabels(String s) {
		int[] last = new int[26];
		int len = s.length();
		for (int i = 0; i < len; i++) {
			last[s.charAt(i) - 'a'] = i;
		}
		List<Integer> partition = new ArrayList<>();
		int start = 0, end = 0;
		for (int i = 0; i < len; i++) {
			end = Math.max(end, last[s.charAt(i) - 'a']);
			if (i == end) {
				partition.add(end - start + 1);
				start = end + 1;
			}
		}
		return partition;
	}

	// 二叉树的深度
	public int maxDepth(TreeNode root) {
		if (root == null) {
			return 0;
		}
		return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
	}

	// 平衡二叉树
	public boolean isBalanced(TreeNode root) {
		if (root == null) {
			return true;
		} else {
			return Math.abs(depth(root.left) - depth(root.right)) <= 1 && isBalanced(root.left) && isBalanced(root.right);
		}
	}

	public int depth(TreeNode root) {
		if (root == null) {
			return 0;
		} else {
			return Math.max(depth(root.left), depth(root.right)) + 1;
		}
	}

	// 七进制数
	public String convertToBase7(int num) {
		boolean flag = num < 0;
		if (flag) {
			num = -num;
		}
		StringBuilder ans = new StringBuilder();
		do {
			ans.append(num % 7);
			num /= 7;
		} while (num != 0);
		return flag ? ans.append("-").reverse().toString() : ans.reverse().toString();
	}

	// 去除重复字母
	public String removeDuplicateLetters(String s) {
		Deque<Character> stack = new ArrayDeque<>();
		int[] count = new int[256];
		for (int i = 0; i < s.length(); i++) {
			count[s.charAt(i)]++;
		}
		boolean[] inStack = new boolean[256];
		for (Character c : s.toCharArray()) {
			count[c]--;
			if (inStack[c]) {
				continue;
			}
			while (!stack.isEmpty() && stack.peek() > c) {
				if (count[stack.peek()] == 0) {
					break;
				}
				inStack[stack.poll()] = false;
			}
			stack.push(c);
			inStack[c] = true;
		}
		StringBuilder sb = new StringBuilder();
		while (!stack.isEmpty()) {
			sb.append(stack.poll());
		}
		return sb.reverse().toString();
	}

	// 第N个斐波那契数
	public int tribonacci(int n) {
		if (n <= 1) {
			return n;
		}
		if (n == 2) {
			return 1;
		}
		int a = 0, b = 1, c = 1;
		for (int i = 3; i <= n; i++) {
			int d = a + b + c;
			a = b;
			b = c;
			c = d;
		}
		return c;
	}

	// 两数之和
	public int[] twoSum(int[] nums, int target) {
		Map<Integer, Integer> record = new HashMap<>();
		for (int i = 0; i < nums.length; i++) {
			if (record.containsKey(target - nums[i])) {
				return new int[]{record.get(target - nums[i]), i};
			}
			record.put(nums[i], i);
		}
		return new int[0];
	}

	// 寻找峰值
	public int findPeakElement(int[] nums) {
		int n = nums.length;
		int l = 0, r = n - 1;
		while (l < r) {
			int mid = l + r >> 1;
			if (nums[mid] > nums[mid + 1]) {
				r = mid;
			} else {
				l = mid + 1;
			}
		}
		return r;
	}

	// 长度最小的子数组
	public int minSubArrayLen(int target, int[] nums) {
		int n = nums.length;
		if (n == 0) {
			return 0;
		}
		int ans = Integer.MAX_VALUE;
		int start = 0, end = 0;
		int sum = 0;
		while (end < n) {
			sum += nums[end];
			while (sum >= target) {
				ans = Math.min(ans, end - start + 1);
				sum -= nums[start];
				start++;
			}
			end++;
		}
		return ans == Integer.MAX_VALUE ? 0 : ans;
	}
}
