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
			cnt += c;
			ans += c * b;
		}
		return ans;
	}

	/**
	 * 全局倒置与局部倒置
	 * 核心:任意一个“局部倒置”均满足“全局倒置”的定义，因此要判定两者数量是否相同，可转换为统计是否存在「不满足“局部倒置”定义的“全局倒置”」
	 */
	public boolean isIdealPermutation(int[] nums) {
		for (int i = 0; i < nums.length; i++) {
			if (Math.abs(nums[i] - i) >= 2) {
				return false;
			}
		}
		return true;
	}

	// 二叉树的右视图
	List<Integer> res = new ArrayList<>();

	public List<Integer> rightSideView(TreeNode root) {
		dfs(root, 0);
		return res;
	}

	private void dfs(TreeNode root, int depth) {
		if (root == null) {
			return;
		}
		if (depth == res.size()) {
			res.add(root.val);
		}
		depth++;
		dfs(root.right, depth);
		dfs(root.left, depth);
	}

	// 翻转单词顺序
	public String reverseWords(String s) {
		s = s.trim();
		int j = s.length() - 1, i = j;
		StringBuilder res = new StringBuilder();
		while (i >= 0) {
			while (i >= 0 && s.charAt(i) != ' ') {
				i--;
			}
			res.append(s.substring(i + 1, j + 1) + " ");
			while (i >= 0 && s.charAt(i) == ' ') {
				i--;
			}
			j = i;
		}
		return res.toString().trim();
	}

	// 找到最高海拔
	public int largestAltitude(int[] gain) {
		int cur = 0, ans = 0;
		for (int x : gain) {
			cur += x;
			ans = Math.max(ans, cur);
		}
		return ans;
	}

	// 二叉树的深度
	public int maxDepth(TreeNode root) {
		if (root == null) {
			return 0;
		}
		return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
	}

	/**
	 * 二进制中1个个数
	 * 若 n & 1 = 0 ，则n二进制最右一位为 0
	 * 若n & 1 = 1,则n二进制最右位为 1
	 */
	public int hammingWeight(int n) {
		int res = 0;
		while (n != 0) {
			res += n & 1;
			n >>>= 1;
		}
		return res;
	}

	// 香槟塔
	public double champagneTower(int poured, int query_row, int query_glass) {
		double[][] f = new double[query_row + 10][query_row + 10];
		f[0][0] = poured;
		for (int i = 0; i <= query_row; i++) {
			for (int j = 0; j <= i; j++) {
				if (f[i][j] <= 1) {
					continue;
				}
				f[i + 1][j] += (f[i][j] - 1) / 2;
				f[i + 1][j + 1] += (f[i][j] - 1) / 2;
			}
		}
		return Math.min(f[query_row][query_glass], 1);
	}

	// 剪绳子
	public int cuttingRope(int n) {
		if (n <= 3) {
			return n - 1;
		}
		int a = n / 3, b = n % 3;
		if (b == 0) {
			return (int) Math.pow(3, a);
		}
		if (b == 1) {
			return (int) Math.pow(3, a - 1) * 4;
		}
		return (int) Math.pow(3, a) * 2;
	}

	// 和为s的连续正数序列
	public int[][] findContinuousSequence(int target) {
		int i = 1, j = 2, s = 3;
		List<int[]> res = new ArrayList<>();
		while (i < j) {
			if (s == target) {
				int[] ans = new int[j - i + 1];
				for (int k = i; k <= j; k++) {
					ans[k - i] = k;
				}
				res.add(ans);
			}
			if (s >= target) {
				s -= i;
				i++;
			} else {
				j++;
				s += j;
			}
		}
		return res.toArray(new int[0][]);
	}

	// 数组中出现次数超过一半的数字--摩尔投票
	public int majorityElement(int[] nums) {
		int x = 0, votes = 0, count = 0;
		for (int num : nums) {
			if (votes == 0) {
				x = num;
			}
			votes += num == x ? 1 : -1;
		}
		return x;
	}

	/**
	 * 数组中数字出现的次数
	 * &:运算法则为遇0得0。也就是说只要有0，结果即为0。
	 * ^:运算法则为相同取0，不同取1。异或运算，关键在异上面，异为1，否则为0
	 */
	public int[] singleNumbers(int[] nums) {
		int z = 0;
		for (int num : nums) {
			z ^= num;
		}
		int m = 1;
		while ((z & m) == 0) {
			m <<= 1;
		}
		int x = 0, y = 0;
		for (int num : nums) {
			if ((num & m) == 0) {
				x ^= num;
			} else {
				y ^= num;
			}
		}
		return new int[]{x, y};
	}

	// 盒子中小球的最大数量
	public int countBalls(int l, int r) {
		int ans = 0;
		int[] cnts = new int[50];
		for (int i = 0; i <= r; i++) {
			int j = i, cur = 0;
			while (j != 0) {
				cur += j % 10;
				j /= 10;
			}
			if (++cnts[cur] > ans) {
				ans = cnts[cur];
			}
		}
		return ans;
	}

	// 区间子数组个数
	public int numSubarrayBoundedMax(int[] nums, int left, int right) {
		int n = nums.length, ans = 0, i0 = -1, i1 = -1;
		for (int i = 0; i < n; ++i) {
			if (nums[i] > right) {
				i0 = i;
			}
			if (nums[i] >= left) {
				i1 = i;
			}
			ans += i1 - i0;
		}
		return ans;
	}

	// 打印从1到n的最大n位数
	int[] printNumbersRes;
	int count = 0;

	public int[] printNumbers(int n) {
		printNumbersRes = new int[(int) Math.pow(10, n) - 1];
		for (int digit = 1; digit < n + 1; digit++) {
			for (char first = '1'; first <= '9'; first++) {
				char[] num = new char[digit];
				num[0] = first;
				dfs(1, num, digit);
			}
		}
		return printNumbersRes;
	}

	private void dfs(int index, char[] num, int digit) {
		if (index == digit) {
			printNumbersRes[count++] = Integer.parseInt(String.valueOf(num));
			return;
		}
		for (char i = '0'; i <= '9'; i++) {
			num[index] = i;
			dfs(index + 1, num, digit);
		}
	}

	// 圆圈中最后剩下的数字
	public int lastRemaining(int n, int m) {
		int ans = 0;
		for (int i = 2; i <= n; i++) {
			ans = (ans + m) % i;
		}
		return ans;
	}

	// 数组形式的整数加法
	public List<Integer> addToArrayForm(int[] num, int k) {
		int n = num.length;
		LinkedList<Integer> res = new LinkedList<>();
		int sum = 0, carry = 0;
		for (int i = n - 1; i >= 0 || k != 0; k = k / 10, i--) {
			int x = i >= 0 ? num[i] : 0;
			int y = k != 0 ? k % 10 : 0;
			sum = x + y + carry;
			carry = sum / 10;
			res.addFirst(sum % 10);
		}
		if (carry != 0) {
			res.add(0, carry);
		}
		return res;
	}

	// 情感丰富的文字
	public int expressiveWords(String s, String[] words) {
		int result = 0;
		char[] sc = s.toCharArray();
		for (String word : words) {
			result += stretchyWord(sc, word.toCharArray()) ? 1 : 0;
		}
		return result;
	}

	private boolean stretchyWord(char[] sc, char[] wtc) {
		if (sc.length < wtc.length) {
			return false;
		}
		int cp, p1 = 0, p2 = p1;
		while ((cp = p1) < sc.length && p2 < wtc.length) {
			int c1 = 0, c2 = 0;
			while (p1 < sc.length && sc[p1] == sc[cp]) {
				c1++;
				p1++;
			}
			while (p2 < wtc.length && wtc[p2] == sc[cp]) {
				c2++;
				p2++;
			}
			if ((c1 != c2 && c1 < 3) || (c1 < c2 && c1 >= 3)) {
				return false;
			}
		}
		return p1 == sc.length && p2 == wtc.length;
	}

	// 删除有序数组中的重复项
	public int removeDuplicates(int[] nums) {
		if (nums == null || nums.length == 0) {
			return 0;
		}
		int p = 0, q = 1;
		while (q < nums.length) {
			if (nums[p] != nums[q]) {
				if (q - p > 1) {
					nums[p + 1] = nums[q];
				}
				p++;
			}
			q++;
		}
		return p + 1;
	}

	// 最后一个单词的长度
	public int lengthOfLastWord(String s) {
		if (s == null || s.length() == 0) {
			return 0;
		}
		int count = 0;
		for (int i = s.length() - 1; i >= 0; i--) {
			if (s.charAt(i) == ' ') {
				if (count == 0) {
					continue;
				}
				break;
			}
			count++;
		}
		return count;
	}

	// 区间列表的交集
	public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
		List<int[]> ans = new ArrayList<>();
		int i = 0, j = 0;
		while (i < firstList.length && j < secondList.length) {
			int low = Math.max(firstList[i][0], secondList[j][0]);
			int high = Math.min(firstList[i][1], secondList[j][1]);
			if (low <= high) {
				ans.add(new int[]{low, high});
			}
			if (firstList[i][1] < secondList[j][1]) {
				i++;
			} else {
				j++;
			}
		}
		return ans.toArray(new int[ans.size()][]);
	}

	// 等差数列划分,n是等差数列的长度，子等差数列数等于 (n-2)(n-1)/2
	public int numberOfArithmeticSlices(int[] nums) {
		int len = nums.length;
		if (len < 3) {
			return 0;
		}
		int preDiff = nums[1] - nums[0];
		int l = 2;
		int res = 0;
		for (int i = 2; i < len; i++) {
			int diff = nums[i] - nums[i - 1];
			if (diff == preDiff) {
				l++;
			} else {
				res += (l - 1) * (l - 2) / 2;
				l = 2;
				preDiff = diff;
			}
		}
		res += (l - 1) * (l - 2) / 2;
		return res;
	}

	// 生成交替二进制字符串额最少操作数
	public int minOperations(String s) {
		int n = s.length(), count = 0;
		for (int i = 0; i < n; i++) {
			count += (s.charAt(i) - '0') ^ (i & 1);
		}
		return Math.min(count, n - count);
	}

}
