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


	// 删除排序链表中的重复元素Ⅱ
	public ListNode deleteDuplicates(ListNode head) {
		if (head == null || head.next == null) {
			return head;
		}
		if (head.val != head.next.val) {
			head.next = deleteDuplicates(head.next);
			return head;
		} else {
			ListNode tmp = head.next.next;
			while (tmp != null && tmp.val == head.val) {
				tmp = tmp.next;
			}
			return deleteDuplicates(tmp);
		}
	}


	// 二叉树的前序遍历
	public List<Integer> preorderTraversal(TreeNode root) {
		List<Integer> res = new ArrayList<>();
		preOrder(root, res);
		return res;
	}

	private void preOrder(TreeNode root, List<Integer> res) {
		if (root == null) {
			return;
		}
		res.add(root.val);
		preOrder(root.left, res);
		preOrder(root.right, res);
	}

	// 存在重复元素二
	public boolean containsNearbyDuplicate(int[] nums, int k) {
		int n = nums.length;
		Set<Integer> set = new HashSet<>();
		for (int i = 0; i < n; i++) {
			if (i > k) {
				set.remove(nums[i - k - 1]);
			}
			if (set.contains(nums[i])) {
				return true;
			}
			set.add(nums[i]);
		}
		return false;
	}

	//  找到最近的有相同 X 或 Y 坐标的点
	public int nearestValidPoint(int x, int y, int[][] points) {
		int ans = -1, min = Integer.MAX_VALUE;
		for (int i = 0; i < points.length; i++) {
			int a = points[i][0], b = points[i][1];
			if (a == x || b == y) {
				int cnt = Math.abs(a - x) + Math.abs(b - y);
				if (cnt < min) {
					min = cnt;
					ans = i;
				}
			}
		}
		return ans;
	}

	// 存在重复元素Ⅲ
	long size;

	public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
		int n = nums.length;
		Map<Long, Long> map = new HashMap<>();
		size = t + 1L;
		for (int i = 0; i < n; i++) {
			long u = nums[i] * 1L;
			long idx = getIdx(u);
			if (map.containsKey(idx)) {
				return true;
			}
			long l = idx - 1, r = idx + 1;
			if (map.containsKey(l) && u - map.get(l) <= t) {
				return true;
			}
			if (map.containsKey(r) && map.get(r) - u <= t) {
				return true;
			}
			map.put(idx, u);
			if (i >= k) {
				map.remove(getIdx(nums[i - k] * 1L));
			}
		}
		return false;
	}

	private long getIdx(long u) {
		return u >= 0 ? u / size : ((u + 1) / size) - 1;
	}

	// 旋转函数
	public int maxRotateFunction(int[] nums) {
		int n = nums.length;
		int[] sum = new int[2 * n + 10];
		for (int i = 1; i <= 2 * n; i++) {
			sum[i] = sum[i - 1] + nums[(i - 1) % n];
		}
		int ans = 0;
		for (int i = 1; i <= n; i++) {
			ans += nums[i - 1] * (i - 1);
		}
		for (int i = n + 1, cur = ans; i < 2 * n; i++) {
			cur += nums[(i - 1) % n] * (n - 1);
			cur -= sum[i - 1] - sum[i - n];
			if (cur > ans) {
				ans = cur;
			}
		}
		return ans;
	}

	// 替换后的最长重复子串
	public int characterReplacement(String s, int k) {
		if (s == null) {
			return 0;
		}
		int[] map = new int[26];
		char[] chars = s.toCharArray();
		int left = 0;
		int right = 0;
		int maxHis = 0;
		for (right = 0; right < chars.length; right++) {
			int index = chars[right] - 'A';
			map[index]++;
			maxHis = Math.max(maxHis, map[index]);
			if (right - left + 1 > maxHis + k) {
				map[chars[left] - 'A']--;
				left++;
			}
		}
		return chars.length - left;
	}

	// 字符串中第二大的数字
	public int secondHighest(String s) {
		int a = -1, b = -1;
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (Character.isDigit(c)) {
				int v = c - '0';
				if (v > a) {
					b = a;
					a = v;
				} else if (v > b && v < a) {
					b = v;
				}
			}
		}
		return b;
	}

	// k个不同整数的子数组
	public int subarraysWithKDistinct(int[] nums, int k) {
		int n = nums.length;
		int[] lower = new int[n], upper = new int[n];
		find(lower, nums, k);
		find(upper, nums, k - 1);
		int ans = 0;
		for (int i = 0; i < n; i++) {
			ans += upper[i] - lower[i];
		}
		return ans;
	}

	private void find(int[] arr, int[] nums, int k) {
		int n = nums.length;
		int[] cnt = new int[n + 1];
		for (int i = 0, j = 0, sum = 0; i < n; i++) {
			int right = nums[i];
			if (cnt[right] == 0) {
				sum++;
			}
			cnt[right]++;
			while (sum > k) {
				int left = nums[j++];
				cnt[left]--;
				if (cnt[left] == 0) {
					sum--;
				}
			}
			arr[i] = j;
		}
	}

	// 最接近目标的甜点成本
	public int closestCost(int[] baseCosts, int[] toppingCosts, int target) {
		int min = (Arrays.stream(baseCosts).min().getAsInt());
		if (min >= target) {
			return min;
		}
		int upper = 2 * target - min;
		boolean[] dp = new boolean[upper];
		for (int base : baseCosts) {
			if (base < upper) {
				dp[base] = true;
			}
		}
		for (int top : toppingCosts) {
			for (int j = upper - 1; j >= min; j--) {
				if (dp[j] && (j + top) < upper) {
					dp[j + top] = true;
				}
				if (dp[j] && (j + 2 * top) < upper) {
					dp[j + 2 * top] = true;
				}
			}
		}
		int ans = min;
		for (int i = min + 1; i < upper; ++i) {
			if (dp[i]) {
				if (Math.abs(i - target) < Math.abs(ans - target)) {
					ans = i;
				} else if (Math.abs(i - target) == Math.abs(ans - target)) {
					ans = Math.min(ans, i);
				}
			}
		}
		return ans;
	}

	// 最大子数组和
	public int maxSubArray(int[] nums) {
		int pre = 0;
		int res = nums[0];
		for (int num : nums) {
			pre = Math.max(pre + num, num);
			res = Math.max(res, pre);
		}
		return res;
	}


	// 串联所有单词的子串
	public List<Integer> findSubstring(String s, String[] words) {
		int n = s.length(), m = words.length, w = words[0].length();
		Map<String, Integer> map = new HashMap<>();
		for (String word : words) {
			map.put(word, map.getOrDefault(word, 0) + 1);
		}
		List<Integer> ans = new ArrayList<>();
		out:
		for (int i = 0; i + m * w <= n; i++) {
			Map<String, Integer> cur = new HashMap<>();
			String sub = s.substring(i, i + m * w);
			for (int j = 0; j < sub.length(); j += w) {
				String item = sub.substring(j, j + w);
				if (!map.containsKey(item)) {
					continue out;
				}
				cur.put(item, cur.getOrDefault(item, 0) + 1);
			}
			if (cur.equals(map)) {
				ans.add(i);
			}
		}
		return ans;
	}

	// 分隔回文串
	public List<List<String>> partition(String s) {
		int len = s.length();
		List<List<String>> res = new ArrayList<>();
		if (len == 0) {
			return res;
		}
		char[] chars = s.toCharArray();
		boolean[][] dp = new boolean[len][len];
		for (int right = 0; right < len; right++) {
			for (int left = 0; left <= right; left++) {
				if (chars[left] == chars[right] && (right - left <= 2 || dp[left + 1][right - 1])) {
					dp[left][right] = true;
				}
			}
		}
		Deque<String> deque = new ArrayDeque<>();
		dfs(s, 0, len, dp, deque, res);
		return res;
	}

	private void dfs(String s, int index, int len, boolean[][] dp, Deque<String> deque, List<List<String>> res) {
		if (index == len) {
			res.add(new ArrayList<>(deque));
			return;
		}
		for (int i = index; i < len; i++) {
			if (dp[index][i]) {
				deque.addLast(s.substring(index, i + 1));
				dfs(s, i + 1, len, dp, deque, res);
				deque.removeLast();
			}
		}
	}

	// 字符串中不同整数的数目
	public int numDifferentIntegers(String word) {
		Set<String> set = new HashSet<>();
		for (int i = 0; i < word.length(); i++) {
			if (word.charAt(i) <= '9') {
				int j = i;
				while (j < word.length() && word.charAt(j) <= '9') {
					j++;
				}
				while (i < j && word.charAt(i) == '0') {
					i++;
				}
				set.add(word.substring(i, j));
				i = j;
			}
		}
		return set.size();
	}

	// 复原IP地址
	public List<String> restoreIpAddresses(String s) {
		int len = s.length();
		List<String> res = new ArrayList<>();
		if (len > 12 || len < 4) {
			return res;
		}
		Deque<String> path = new ArrayDeque<>();
		dfs(s, len, 0, 4, path, res);
		return res;
	}

	private void dfs(String s, int len, int begin, int residue, Deque<String> path, List<String> res) {
		if (begin == len) {
			if (residue == 0) {
				res.add(String.join(".", path));
			}
			return;
		}
		for (int i = begin; i < begin + 3; i++) {
			if (i >= len) {
				break;
			}
			if (residue * 3 < len - i) {
				continue;
			}
			if (judgeIpSegment(s, begin, i)) {
				String cur = s.substring(begin, i + 1);
				path.addLast(cur);
				dfs(s, len, i + 1, residue - 1, path, res);
				path.removeLast();
			}
		}
	}

	private boolean judgeIpSegment(String s, int left, int right) {
		int len = right - left + 1;
		if (len > 1 && s.charAt(left) == '0') {
			return false;
		}
		int res = 0;
		while (left <= right) {
			res = res * 10 + s.charAt(left) - '0';
			left++;
		}
		return res >= 0 && res <= 255;
	}

	// 通过最少操作次数使数组的和相等
	public int minOperations(int[] nums1, int[] nums2) {
		if (6 * nums1.length < nums2.length || 6 * nums2.length < nums1.length) {
			return -1;
		}
		int d = Arrays.stream(nums2).sum() - Arrays.stream(nums1).sum();
		if (d < 0) {
			d = -d;
			int[] tmp = nums1;
			nums1 = nums2;
			nums2 = tmp;
		}
		int[] cnt = new int[6];
		for (int x : nums1) {
			++cnt[6 - x];
		}
		for (int x : nums2) {
			++cnt[x - 1];
		}
		for (int i = 5, ans = 0; ; --i) {
			if (i * cnt[i] >= d) {
				return ans + (d + i - 1) / i;
			}
			ans += cnt[i];
			d -= i * cnt[i];
		}
	}


	// 判断国际象棋棋盘中一个格子的颜色
	public boolean squareIsWhite(String coordinates) {
		return (coordinates.charAt(0) + coordinates.charAt(1)) % 2 == 1;
	}

	// 划分为k个相等的子集
	public boolean canPartitionKSubsets(int[] nums, int k) {
		int sum = 0;
		for (int i = 0; i < nums.length; i++) {
			sum += nums[i];
		}
		if (sum % k != 0) {
			return false;
		}
		int target = sum / k;
		Arrays.sort(nums);
		int l = 0, r = nums.length - 1;
		while (l <= r) {
			int temp = nums[l];
			nums[l] = nums[r];
			nums[r] = temp;
			l++;
			r--;
		}
		return backTrace(nums, 0, new int[k], k, target);
	}

	private boolean backTrace(int[] nums, int index, int[] bucket, int k, int target) {
		if (index == nums.length) {
			return true;
		}
		for (int i = 0; i < k; i++) {
			if (i > 0 && bucket[i] == bucket[i - 1]) {
				continue;
			}
			if (bucket[i] + nums[index] > target) {
				continue;
			}
			bucket[i] += nums[index];
			if (backTrace(nums, index + 1, bucket, k, target)) {
				return true;
			}
			bucket[i] -= nums[index];
		}
		return false;
	}

	// 判断一个数字是否可以表示成三的幂的和
	public boolean checkPowersOfThree(int n) {
		while (n != 0) {
			if (n % 3 == 2) {
				return false;
			}
			n /= 3;
		}
		return true;
	}

	// 二维网格迁移
	public List<List<Integer>> shiftGrid(int[][] grid, int k) {
		int n = grid.length, m = grid[0].length;
		int[][] mat = new int[n][m];
		for (int i = 0; i < m; i++) {
			int tcol = (i + k) % m, trow = ((i + k) / m) % n, idx = 0;
			while (idx != n) {
				mat[(trow++ % n)][tcol] = grid[idx++][i];
			}
		}
		List<List<Integer>> ans = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			List<Integer> alist = new ArrayList<>();
			for (int j = 0; j < m; j++) {
				alist.add(mat[i][j]);
			}
			ans.add(alist);
		}
		return ans;
	}

	// 找出缺失的观测数据
	// 以sum/n为基数，sum%n为偏差diff，如果diff>0，则取1加到基数上，否则使用基数即可；
	public int[] missingRolls(int[] rolls, int mean, int n) {
		int m = rolls.length, cnt = m + n;
		int t = mean * cnt;
		for (int i : rolls) {
			t -= i;
		}
		if (t < n || t > 6 * n) {
			return new int[0];
		}
		int[] ans = new int[n];
		Arrays.fill(ans, t / n);
		int diff = t - (t / n * n), idx = 0;
		while (diff-- > 0) {
			ans[idx++]++;
		}
		return ans;
	}

	// 增减字符串匹配
	public int[] diStringMatch(String s) {
		int n = s.length(), l = 0, r = n, idx = 0;
		int[] ans = new int[n + 1];
		for (int i = 0; i < n; i++) {
			ans[idx++] = s.charAt(i) == 'I' ? l++ : r--;
		}
		ans[idx] = l;
		return ans;
	}

}
