package daily;

import java.util.*;

/**
 * @author Kezi
 * @date 2022年07月23日 22:39
 */
public class SolutionTwo {

	// 公交站台的距离
	public int distanceBetweenBusStops(int[] dist, int s, int t) {
		int n = dist.length, i = s, j = s, a = 0, b = 0;
		while (i != t) {
			a += dist[i];
			if (++i == n) {
				i = 0;
			}
		}
		while (j != t) {
			if (--j < 0) {
				j = n - 1;
			}
			b += dist[t];
		}
		return Math.min(a, b);
	}

	// 出现频率最高的K个数字
	public int[] topKFrequent(int[] nums, int k) {
		Map<Integer, Integer> map = new HashMap<>();
		for (int num : nums) {
			map.put(num, map.getOrDefault(num, 0) + 1);
		}
		// 使用优先队列
		PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);
		for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
			int num = entry.getKey(), cnt = entry.getValue();
			pq.offer(new int[]{num, cnt});
			if (pq.size() > k) {
				pq.poll();
			}
		}
		int[] ans = new int[pq.size()];
		for (int i = 0; i < k; i++) {
			ans[i] = pq.poll()[0];
		}
		return ans;
	}

	// 山峰数组额顶部
	public int peakIndexInMountainArray(int[] arr) {
		int n = arr.length;
		int left = 1, right = n - 2, ans = 0;
		while (left <= right) {
			int mid = (left + right) / 2;
			if (arr[mid] > arr[mid + 1]) {
				ans = mid;
				right = mid - 1;
			} else {
				left = mid + 1;
			}
		}
		return ans;
	}

	// 排序数组中只出现一次的数字
	public int singleNonDuplicate(int[] nums) {
		int n = nums.length, l = 0, r = n - 1;
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

	/**
	 * 有效的正方形
	 * 该图形是正方形，那么任意三点组成的一定是等腰直角三角形，用此条件作为判断
	 */
	long len = -1;

	public boolean validSquare(int[] p1, int[] p2, int[] p3, int[] p4) {
		return calc(p1, p2, p3) && calc(p1, p2, p4) && calc(p1, p3, p4) && calc(p2, p3, p4);
	}

	boolean calc(int[] a, int[] b, int[] c) {
		long l1 = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
		long l2 = (a[0] - c[0]) * (a[0] - c[0]) + (a[1] - c[1]) * (a[1] - c[1]);
		long l3 = (b[0] - c[0]) * (b[0] - c[0]) + (b[1] - c[1]) * (b[1] - c[1]);
		boolean ok = (l1 == l2 && l1 + l2 == l3) || (l1 == l3 && l1 + l3 == l2) || (l2 == l3 && l2 + l3 == l1);
		if (!ok) {
			return false;
		}
		if (len == -1) {
			len = Math.min(l1, l2);
		} else if (len == 0 || len != Math.min(l1, l2)) {
			return false;
		}
		return true;
	}

	// 链表的中间节点
	public ListNode middleNode(ListNode head) {
		ListNode slow = head, fast = head;
		while (fast != null && fast.next != null) {
			slow = slow.next;
			fast = fast.next.next;
		}
		return slow;
	}

	// 环形链表Ⅱ
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
		while (slow != fast) {
			slow = slow.next;
			fast = fast.next;
		}
		return fast;
	}

	// 层内元素之和
	public int maxLevelSum(TreeNode root) {
		int ans = 1, maxSum = root.val;
		List<TreeNode> q = new ArrayList<>();
		q.add(root);
		for (int lev = 1; !q.isEmpty(); ++lev) {
			List<TreeNode> nq = new ArrayList<>();
			int sum = 0;
			for (TreeNode node : q) {
				sum += node.val;
				if (node.left != null) {
					nq.add(node.left);
				}
				if (node.right != null) {
					nq.add(node.right);
				}
			}
			if (sum > maxSum) {
				maxSum = sum;
				ans = lev;
			}
			q = nq;
		}
		return ans;
	}

	// 生成每种字符串都是奇数个的字符串
	public String generateTheString(int n) {
		StringBuilder sb = new StringBuilder();
		if (n % 2 == 0 && --n >= 0) {
			sb.append("a");
		}
		while (n-- > 0) {
			sb.append("b");
		}
		return sb.toString();
	}

	// 分割字符串的最大得分
	public int maxScore(String s) {
		int n = s.length(), cur = s.charAt(0) == '0' ? 1 : 0;
		for (int i = 1; i < n; i++) {
			cur += s.charAt(i) - '0';
		}
		int ans = cur;
		for (int i = 1; i < n - 1; i++) {
			cur += s.charAt(i) == '0' ? 1 : -1;
			ans = Math.max(ans, cur);
		}
		return ans;
	}

	// 反转链表
	public ListNode reverseList(ListNode head) {
		ListNode pre = null;
		ListNode cur = head;
		while (cur != null) {
			ListNode next = cur.next;
			cur.next = pre;
			pre = cur;
			cur = next;
		}
		return pre;
	}


	// 回文数字
	public boolean isPalindrome(int x) {
		if (x < 0 || (x % 10 == 0 && x != 0)) {
			return false;
		}
		int rev = 0;
		while (x > rev) {
			rev = rev * 10 + x % 10;
			x /= 10;
		}
		return x == rev || x == rev / 10;
	}

	/**
	 * 分割平衡字符串
	 * 更好的方式是转换为数学判定，使用 1 来代指 L 得分，使用 -1 来代指 R 得分
	 * 题目要求分割的 LR 子串尽可能多，直观上应该是尽可能让每个分割串尽可能短(归纳法证明该猜想的正确性)
	 */
	public int balancedStringSplit(String s) {
		char[] cs = s.toCharArray();
		int n = cs.length;
		int ans = 0;
		for (int i = 0; i < n; ) {
			int j = i + 1, score = cs[i] == 'L' ? 1 : -1;
			while (j < n && score != 0) {
				score += cs[j++] == 'L' ? 1 : -1;
			}
			i = j;
			ans++;
		}
		return ans;
	}

	// 数组中第K大的元素
	public int findKthLargest(int[] nums, int k) {
		PriorityQueue<Integer> pq = new PriorityQueue<>(k, Comparator.comparingInt(a -> a));
		for (int num : nums) {
			pq.offer(num);
			if (pq.size() > k) {
				pq.poll();
			}
		}
		return pq.peek();
	}

	// 一年中的第几天
	static int[] nums = new int[]{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
	static int[] f = new int[13];

	static {
		for (int i = 1; i <= 12; i++) {
			f[i] = f[i - 1] + nums[i - 1];
		}
	}

	public int dayOfYear(String date) {
		String[] ss = date.split("-");
		int y = Integer.parseInt(ss[0]), m = Integer.parseInt(ss[1]), d = Integer.parseInt(ss[2]);
		boolean isLeap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
		int ans = m > 2 && isLeap ? f[m - 1] + 1 : f[m - 1];
		return ans + d;
	}

	// 含有k个元素的组合
	List<List<Integer>> ans;

	public List<List<Integer>> combine(int n, int k) {
		ans = new ArrayList<>();
		dfs(n, 1, k, new ArrayList<>());
		return ans;
	}

	private void dfs(int n, int i, int k, List<Integer> list) {
		// 剪枝
		if (n - i + 1 < k) {
			return;
		}
		if (k == 0) {
			ans.add(new ArrayList<>(list));
			return;
		}
		list.add(i);
		dfs(n, i + 1, k - 1, list);
		list.remove(list.size() - 1);
		dfs(n, i + 1, k, list);
	}

	// 找到K个最接近的元素
	public List<Integer> findClosestElements(int[] arr, int k, int x) {
		int n = arr.length, l = 0, r = n - 1;
		while (l < r) {
			int mid = l + r + 1 >> 1;
			if (arr[mid] <= x) {
				l = mid;
			} else {
				r = mid - 1;
			}
		}
		r = r + 1 < n && Math.abs(arr[r + 1] - x) < Math.abs(arr[r] - x) ? r + 1 : r;
		int i = r - 1, j = r + 1;
		while (j - i - 1 < k) {
			if (i >= 0 && j < n) {
				if (Math.abs(arr[j] - x) < Math.abs(arr[i] - x)) {
					j++;
				} else {
					i--;
				}
			} else if (i >= 0) {
				i--;
			} else {
				j++;
			}
		}
		List<Integer> ans = new ArrayList<>();
		for (int p = i + 1; p <= j - 1; p++) {
			ans.add(arr[p]);
		}
		return ans;
	}

	// 重新排列数组
	public int[] shuffle(int[] nums, int n) {
		int[] ans = new int[2 * n];
		for (int i = 0, j = n, k = 0; k < 2 * n; n++) {
			ans[k] = k % 2 == 0 ? nums[i++] : nums[j++];
		}
		return ans;
	}

	// 最大二叉树Ⅱ
	// 懂了又好像没懂
	public TreeNode insertIntoMaxTree(TreeNode root, int val) {
		TreeNode node = new TreeNode(val);
		TreeNode prev = null, cur = root;
		while (cur != null && cur.val > val) {
			prev = cur;
			cur = cur.right;
		}
		if (prev == null) {
			node.left = cur;
			return node;
		} else {
			prev.right = node;
			node.left = cur;
			return root;
		}
	}

	// 验证栈序列
	public boolean validateStackSequences(int[] pushed, int[] popped) {
		Deque<Integer> stack = new ArrayDeque<>();
		int n = pushed.length;
		for (int i = 0, j = 0; i < n; i++) {
			stack.push(pushed[i]);
			while (!stack.isEmpty() && stack.peek() == popped[j]) {
				stack.pop();
				j++;
			}
		}
		return stack.isEmpty();
	}

	// 商品折扣后的最终价格--单调栈
	public int[] finalPrices(int[] ps) {
		int n = ps.length;
		int[] ans = new int[n];
		Deque<Integer> deque = new ArrayDeque<>();
		for (int i = 0; i < n; i++) {
			while (!deque.isEmpty() && ps[deque.peekLast()] >= ps[i]) {
				int idx = deque.pollLast();
				ans[idx] = ps[idx] - ps[i];
			}
			deque.addLast(i);
			ans[i] = ps[i];
		}
		return ans;
	}

	// 最长同值路径
	int result = 0;

	public int longestUnivaluePath(TreeNode root) {
		calculate(root);
		return result;
	}

	public int calculate(TreeNode node) {
		if (node == null) return 0;
		int leftValue = calculate(node.left);
		int rightValue = calculate(node.right);
		leftValue = (node.left != null && node.val == node.left.val) ? ++leftValue : 0;
		rightValue = (node.right != null && node.val == node.right.val) ? ++rightValue : 0;
		result = Math.max(result, leftValue + rightValue);
		return Math.max(leftValue, rightValue);
	}

	// 二进制矩阵中的特殊位置
	public int numSpecial(int[][] mat) {
		int n = mat.length, m = mat[0].length, ans = 0;
		int[] r = new int[n], c = new int[m];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				r[i] += mat[i][j];
				c[j] += mat[i][j];
			}
		}
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				if (mat[i][j] == 1 && r[i] == 1 && c[j] == 1) {
					ans++;
				}
			}
		}
		return ans;
	}

	// 寻找重复的子树
	Map<String, Integer> map = new HashMap<>();
	List<TreeNode> res = new ArrayList<>();

	public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
		dfs(root);
		return res;
	}

	private String dfs(TreeNode root) {
		if (root == null) {
			return "";
		}
		String key = new StringBuilder().append(root.val).append(",").append(dfs(root.left)).append(",").append(dfs(root.right)).toString();
		if (map.getOrDefault(key, 0) == 1) {
			res.add(root);
		}
		map.put(key, map.getOrDefault(key, 0) + 1);
		return key;
	}

	// 重新排列单词间的空格
	public String reorderSpaces(String s) {
		int n = s.length(), cnt = 0;
		List<String> list = new ArrayList<>();
		for (int i = 0; i < n; ) {
			if (s.charAt(i) == ' ' && ++i >= 0 && ++cnt >= 0) {
				continue;
			}
			int j = i;
			while (j < n && s.charAt(j) != ' ') {
				j++;
			}
			list.add(s.substring(i, j));
			i = j;
		}
		StringBuilder sb = new StringBuilder();
		int m = list.size(), t = cnt / Math.max(m - 1, 1);
		String k = "";
		while (t-- > 0) {
			k += " ";
		}
		for (int i = 0; i < m; i++) {
			sb.append(list.get(i));
			if (i != m - 1) {
				sb.append(k);
			}
		}
		while (sb.length() != n) {
			sb.append(" ");
		}
		return sb.toString();
	}

	/**
	 * 优美的排列Ⅱ
	 * 思路：我们需要 k + 1k+1 个数来构造出 kk 个差值。因此我们可以先从 11 开始，使用 n - (k + 1)n−(k+1) 个数来直接升序（通过方式一构造出若干个 11），
	 * 然后从 n - kn−k 开始间隔升序排列，按照 nn 开始间隔降序排列，构造出剩下的序列
	 */
	public int[] constructArray(int n, int k) {
		int[] ans = new int[n];
		int t = n - k - 1;
		for (int i = 0; i < t; i++) {
			ans[i] = i + 1;
		}
		for (int i = t, a = n - k, b = n; i < n; ) {
			ans[i++] = a++;
			if (i < n) {
				ans[i++] = b--;
			}
		}
		return ans;
	}

	// 文件夹操作日志收集器
	public int minOperations(String[] logs) {
		int depth = 0;
		for (String string : logs) {
			if (string.equals("../")) {
				depth = Math.max(0, depth - 1);
			} else if (!string.equals("./")) {
				depth++;
			}
		}
		return depth;
	}

	// 最大交换
	public int maximumSwap(int num) {
		List<Integer> list = new ArrayList<>();
		while (num != 0) {
			list.add(num % 10);
			num /= 10;
		}
		int n = list.size(), ans = 0;
		int[] idx = new int[n];
		for (int i = 0, j = 0; i < n; i++) {
			if (list.get(i) > list.get(j)) {
				j = i;
			}
			idx[i] = j;
		}
		for (int i = n - 1; i >= 0; i--) {
			if (!list.get(idx[i]).equals(list.get(i))) {
				int c = list.get(idx[i]);
				list.set(idx[i], list.get(i));
				list.set(i, c);
				break;
			}
		}
		for (int i = n - 1; i >= 0; i--) {
			ans = ans * 10 + list.get(i);
		}
		return ans;
	}

	// 盛最多水的容器
	public int maxArea(int[] height) {
		int i = 0, j = height.length - 1, res = 0;
		while (i < j) {
			res = height[i] < height[j] ? Math.max(res, (j - i) * height[i++]) : Math.max(res, (j - 1) * height[j--]);
		}
		return res;
	}

	// 从前序遍历与中序遍历序列构造二叉树
	private Map<Integer, Integer> indexMap;

	public TreeNode buildTree(int[] preorder, int[] inorder) {
		int n = preorder.length;
		indexMap = new HashMap<>();
		for (int i = 0; i < n; i++) {
			indexMap.put(inorder[i], i);
		}
		return myBuildTree(preorder, inorder, 0, n - 1, 0, n - 1);
	}

	public TreeNode myBuildTree(int[] preorder, int[] inorder, int preorder_left, int preorder_right, int inorder_left, int inorder_right) {
		if (preorder_left > preorder_right) {
			return null;
		}
		int preorder_root = preorder_left;
		int inorder_root = indexMap.get(preorder[preorder_root]);
		TreeNode root = new TreeNode(preorder[preorder_root]);
		int size_left_subtree = inorder_root - inorder_left;
		root.left = myBuildTree(preorder, inorder, preorder_left + 1, preorder_left + size_left_subtree, inorder_left, inorder_root - 1);
		root.right = myBuildTree(preorder, inorder, preorder_left + size_left_subtree + 1, preorder_right, inorder_root + 1, inorder_right);
		return root;
	}

	// 按照频率将数组升序排序
	public int[] frequencySort(int[] nums) {
		int n = nums.length;
		Map<Integer, Integer> map = new HashMap<>();
		for (int num : nums) {
			map.put(num, map.getOrDefault(num, 0) + 1);
		}
		List<int[]> list = new ArrayList<>();
		for (int key : map.keySet()) {
			list.add(new int[]{key, map.get(key)});
		}
		list.sort((a, b) -> {
			return a[1] != b[1] ? a[1] - b[1] : b[0] - a[0];
		});
		int[] ans = new int[n];
		int idx = 0;
		for (int[] info : list) {
			while (info[1]-- > 0) {
				ans[idx++] = info[0];
			}
		}
		return ans;
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
		while (l < r) {
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

	// 能否连接形成数组
	public boolean canFormArray(int[] arr, int[][] pieces) {
		int n = arr.length, m = pieces.length;
		int[] hash = new int[110];
		for (int i = 0; i < m; i++) {
			hash[pieces[i][0]] = i;
		}
		for (int i = 0; i < n; ) {
			int[] cur = pieces[hash[arr[i]]];
			int len = cur.length, idx = 0;
			while (idx < len && cur[idx] == arr[i + idx]) {
				idx++;
			}
			if (idx == len) {
				i += len;
			} else {
				return false;
			}
		}
		return true;
	}

	// 拆炸弹--前缀和
	public int[] decrypt(int[] code, int k) {
		int n = code.length;
		int[] ans = new int[n];
		if (k == 0) {
			return ans;
		}
		int[] sum = new int[n * 2 + 10];
		for (int i = 1; i <= 2 * n; i++) {
			sum[i] += sum[i - 1] + code[(i - 1) % n];
		}
		for (int i = 1; i <= n; i++) {
			if (k < 0) {
				ans[i - 1] = sum[i + n - 1] - sum[i + n + k - 1];
			} else {
				ans[i - 1] = sum[i + k] - sum[i];
			}
		}
		return ans;
	}

	// 旋转数字
	public int rotatedDigits(int n) {
		int ans = 0;
		out:
		for (int i = 1; i <= n; i++) {
			boolean ok = false;
			int x = i;
			while (x != 0) {
				int t = x % 10;
				x /= 10;
				if (t == 2 || t == 5 || t == 6 || t == 9) {
					ok = true;
				} else if (t != 0 && t != 1 && t != 8) {
					continue out;
				}
			}
			if (ok) {
				ans++;
			}
		}
		return ans;
	}

	// 全部开花的最早一天
	public int earliestFullBloom(int[] plantTime, int[] growTime) {
		int n = plantTime.length;
		List<Integer> list = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			list.add(i);
		}
		Collections.sort(list, (o1, o2) -> growTime[o2] - growTime[o1]);
		int ans = 0;
		int pt = 0, gt = 0;
		for (int index : list) {
			pt += plantTime[index];
			gt = pt + growTime[index];
			ans = Math.max(ans, gt);
		}
		return ans;
	}

	// 判断是否互为字符重排
	public boolean CheckPermutation(String s1, String s2) {
		int n = s1.length(), m = s2.length(), tot = 0;
		if (n != m) {
			return false;
		}
		int[] cnts = new int[256];
		for (int i = 0; i < n; i++) {
			if (++cnts[s1.charAt(i)] == 1) {
				System.out.println(cnts[s1.charAt(i)]);
				tot++;
			}
			if (--cnts[s2.charAt(i)] == 0) {
				System.out.println(cnts[s2.charAt(i)]);
				tot--;
			}
		}
		return tot == 0;
	}

	// 第k个数
	public int getKthMagicNumber(int k) {
		int[] ans = new int[k + 1];
		ans[1] = 1;
		for (int i3 = 1, i5 = 1, i7 = 1, idx = 2; idx <= k; idx++) {
			int a = ans[i3] * 3,b = ans[i5] * 5,c = ans[i7] * 7;
			int min = Math.min(a, Math.min(b,c));
			if (min == a){
				i3++;
			}
			if (min == b){
				i5++;
			}
			if (min == c){
				i7++;
			}
			ans[idx] = min;
		}
		return ans[k];
	}
} 