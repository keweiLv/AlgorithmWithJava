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
			if (nums[i] > 0) {
				return ans;
			}
			if (i > 0 && nums[i] == nums[i - 1]) {
				continue;
			}
			int cur = nums[i];
			int l = i + 1, r = len - 1;
			while (l < r) {
				int tmp = cur + nums[l] + nums[r];
				if (tmp == 0) {
					List<Integer> list = new ArrayList<>();
					list.add(cur);
					list.add(nums[l]);
					list.add(nums[r]);
					ans.add(list);
					while (l < r && nums[l + 1] == nums[l]) {
						++l;
					}
					while (l < r && nums[r - 1] == nums[r]) {
						--r;
					}
					++l;
					--r;
				} else if (tmp < 0) {
					++l;
				} else {
					--r;
				}
			}
		}
		return ans;
	}

	// 岛屿数量
	public int numIslands(char[][] grid) {
		int count = 0;
		for (int i = 0; i < grid.length; i++) {
			for (int j = 0; j < grid[0].length; j++) {
				if (grid[i][j] == '1') {
					dfs(grid, i, j);
					count++;
				}
			}
		}
		return count;
	}

	private void dfs(char[][] grid, int i, int j) {
		if (i < 0 || j < 0 || i > grid.length || j > grid[0].length || grid[i][j] == '0') {
			return;
		}
		grid[i][j] = '0';
		dfs(grid, i + 1, j);
		dfs(grid, i, j + 1);
		dfs(grid, i - 1, j);
		dfs(grid, i, j - 1);
	}

	// 二叉树的最近公共祖先
	public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
		if (root == null || root == p || root == q) {
			return root;
		}
		TreeNode left = lowestCommonAncestor(root.left, p, q);
		TreeNode right = lowestCommonAncestor(root.right, p, q);
		if (left == null) {
			return right;
		}
		if (right == null) {
			return left;
		}
		return root;
	}

	// 判断字符串的两半是否相似
	public boolean halvesAreAlike(String s) {
		Set<Character> set = new HashSet<>();
		for (char c : "aeiouAEIOU".toCharArray()) {
			set.add(c);
		}
		int n = s.length(), count = 0;
		for (int i = 0; i < n; i++) {
			if (!set.contains(s.charAt(i))) {
				continue;
			}
			count += i < n / 2 ? 1 : -1;
		}
		return count == 0;
	}

	// 自定义字符串排序
	public String customSortString(String order, String s) {
		int[] cnts = new int[26];
		for (char c : s.toCharArray()) {
			cnts[c - 'a']++;
		}
		StringBuilder sb = new StringBuilder();
		for (char c : order.toCharArray()) {
			while (cnts[c - 'a']-- > 0) {
				sb.append(c);
			}
		}
		for (int i = 0; i < 26; i++) {
			while (cnts[i]-- > 0) {
				sb.append((char) (i + 'a'));
			}
		}
		return sb.toString();
	}

	// 二叉树的镜像
	public TreeNode mirrorTree(TreeNode root) {
		if (root == null) {
			return null;
		}
		TreeNode leftRoot = mirrorTree(root.right);
		TreeNode rightRoot = mirrorTree(root.left);
		root.left = leftRoot;
		root.right = rightRoot;
		return root;
	}

	// 斐波那契数列
	public int fib(int n) {
		int a = 0, b = 1, sum;
		for (int i = 0; i < n; i++) {
			sum = (a + b) % 1000000007;
			a = b;
			b = sum;
		}
		return a;
	}

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

	// 卡车上的最大单元数
	public int maximumUnits(int[][] boxTypes, int truckSize) {
		int n = boxTypes.length, ans = 0;
		Arrays.sort(boxTypes, (a, b) -> b[1] - a[1]);
		for (int i = 0, cnt = 0; i < n && cnt < truckSize; i++) {
			int a = boxTypes[i][0], b = boxTypes[i][1], c = Math.min(a, truckSize - cnt);
			cnt += c;ans += c * b;
		}
		return ans;
	}
}
