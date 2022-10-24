package forjob;

import java.util.*;

/**
 * @author Kezi
 * @date 2022年10月10日 23:28
 */
public class Solution {
	//上下翻转二叉树
	public TreeNode upsideDownBinaryTree(TreeNode root) {
		TreeNode parent = null, parent_right = null;
		while (root != null) {
			TreeNode root_left = root.left;
			TreeNode root_right = root.right;
			root.left = parent_right;
			root.right = parent;
			parent = root;
			root = root_left;
			parent_right = root_right;
		}
		return parent;
	}

	//  仅执行一次字符串交换能否使两个字符串相等
	public boolean areAlmostEqual(String s1, String s2) {
		int n = s1.length(), a = -1, b = -1;
		for (int i = 0; i < n; i++) {
			if (s1.charAt(i) == s2.charAt(i)) {
				continue;
			}
			if (a == -1) {
				a = i;
			} else if (b == -1) {
				b = i;
			} else {
				return false;
			}
		}
		if (a == -1) {
			return true;
		}
		if (b == -1) {
			return false;
		}
		return s1.charAt(a) == s2.charAt(b) && s1.charAt(b) == s2.charAt(a);
	}

	// 链表组件
	public int numComponents(ListNode head, int[] nums) {
		int ans = 0;
		Set<Integer> set = new HashSet<>();
		for (int num : nums) {
			set.add(num);
		}
		while (head != null) {
			if (set.contains(head.val)) {
				while (head != null && set.contains(head.val)) {
					head = head.next;
				}
				ans++;
			} else {
				head = head.next;
			}
		}
		return ans;
	}

	// 二叉树的层序遍历
	public List<List<Integer>> levelOrder(TreeNode root) {
		List<List<Integer>> res = new ArrayList<>();
		Queue<TreeNode> queue = new ArrayDeque<>();
		if (root != null) {
			queue.add(root);
		}
		while (!queue.isEmpty()) {
			int n = queue.size();
			List<Integer> level = new ArrayList<>();
			for (int i = 0; i < n; i++) {
				TreeNode node = queue.poll();
				level.add(node.val);
				if (node.left != null) {
					queue.add(node.left);
				}
				if (node.right != null) {
					queue.add(node.right);
				}
			}
			res.add(level);
		}
		return res;
	}

	// 最多能完成排序的块
	public int maxChunksToSorted(int[] arr) {
		int n = arr.length, ans = 0;
		for (int i = 0, j = 0, min = n, max = -1; i < n; i++) {
			min = Math.min(min, arr[i]);
			max = Math.max(max, arr[i]);
			if (j == min && i == max) {
				ans++;
				j = i + 1;
				min = n;
				max = -1;
			}
		}
		return ans;
	}

	// 全排列
	public List<List<Integer>> permute(int[] nums) {
		List<List<Integer>> res = new ArrayList<>();
		int len = nums.length;
		boolean[] used = new boolean[len];
		Deque<Integer> path = new ArrayDeque<>(len);
		dfs(nums, len, 0, path, used, res);
		return res;
	}

	private void dfs(int[] nums, int len, int depth, Deque<Integer> path, boolean[] used, List<List<Integer>> res) {
		if (depth == len) {
			res.add(new ArrayList<>(path));
			return;
		}
		for (int i = 0; i < len; i++) {
			if (!used[i]) {
				path.addLast(nums[i]);
				used[i] = true;
				dfs(nums, len, depth + 1, path, used, res);
				used[i] = false;
				path.removeLast();
			}
		}
	}

	// 跳跃游戏Ⅱ
	public int jump(int[] nums) {
		int end = 0;
		int maxPosition = 0;
		int steps = 0;
		for (int i = 0; i < nums.length - 1; i++) {
			maxPosition = Math.max(maxPosition, nums[i] + i);
			if (i == end) {
				end = maxPosition;
				steps++;
			}
		}
		return steps;
	}


	// 链表随机节点
	ListNode head;
	Random random;

	public Solution(ListNode head) {
		this.head = head;
		random = new Random();
	}

	public int getRandom() {
		int i = 1, ans = 0;
		for (ListNode node = head; node != null; node = node.next) {
			if (random.nextInt(i) == 0) {
				ans = node.val;
			}
			++i;
		}
		return ans;
	}

	// 水果成篮
	public int totalFruit(int[] fs) {
		int n = fs.length, ans = 0;
		int[] cnts = new int[n + 10];
		for (int i = 0, j = 0, tot = 0; i < n; i++) {
			if (++cnts[fs[i]] == 1) {
				tot++;
			}
			while (tot > 2) {
				if (--cnts[fs[j++]] == 0) {
					tot--;
				}
			}
			ans = Math.max(ans, i - j + 1);
		}
		return ans;
	}

	// 寻找二叉树的叶子节点
	private Map<Integer, List<Integer>> map = new HashMap<>();

	public List<List<Integer>> findLeaves(TreeNode root) {
		dfs(root);
		return new LinkedList<>(map.values());
	}

	private int dfs(TreeNode root) {
		if (root == null) {
			return 0;
		}
		int leftDistance = dfs(root.left);
		int rightDistance = dfs(root.right);
		int currentDistance = Math.max(leftDistance, rightDistance) + 1;
		map.computeIfAbsent(currentDistance, i -> new LinkedList<>()).add(root.val);
		return currentDistance;
	}


	// 无法吃午餐的学生数量
	public int countStudents(int[] a, int[] b) {
		int[] cnts = new int[2];
		for (int x : a) {
			cnts[x]++;
		}
		for (int i = 0; i < b.length; i++) {
			if (--cnts[b[i]] == -1) {
				return b.length - i;
			}
		}
		return 0;
	}

	// 两个数组的交集
	public int[] intersection(int[] nums1, int[] nums2) {
		if (nums1 == null || nums2.length == 0 || nums2 == null || nums2.length == 0) {
			return new int[0];
		}
		Set<Integer> set1 = new HashSet<>();
		Set<Integer> resSet = new HashSet<>();
		for (int num : nums1) {
			set1.add(num);
		}
		for (int num : nums2) {
			if (set1.contains(num)) {
				resSet.add(num);
			}
		}
		return resSet.stream().mapToInt(x -> x).toArray();
	}

	// 第K个语法符号
	public int kthGrammar(int n, int k) {
		return dfs(n, k, 1) == 0 ? 1 : 0;
	}

	private int dfs(int n, int k, int cur) {
		if (n == 1) {
			return cur;
		}
		if ((k % 2 == 0 && cur == 0) || (k % 2 == 1 && cur == 1)) {
			return dfs(n - 1, (k - 1) / 2 + 1, 1);
		} else {
			return dfs(n - 1, (k - 1) / 2 + 1, 0);
		}
	}

	// 二叉树最大宽度
	Map<Integer, Integer> widthOfBinaryTreeMap = new HashMap<>();
	int ans;

	public int widthOfBinaryTree(TreeNode root) {
		dfs(root, 1, 0);
		return ans;
	}

	private void dfs(TreeNode root, int u, int depth) {
		if (root == null) {
			return;
		}
		if (!widthOfBinaryTreeMap.containsKey(depth)) {
			widthOfBinaryTreeMap.put(depth, u);
		}
		ans = Math.max(ans, u - widthOfBinaryTreeMap.get(depth) + 1);
		u = u - widthOfBinaryTreeMap.get(depth) + 1;
		dfs(root.left, u << 1, depth + 1);
		dfs(root.right, u << 1 | 1, depth + 1);
	}

	// 一次编辑
	public boolean oneEditAway(String a, String b) {
		int n = a.length(), m = b.length();
		if (Math.abs(n - m) > 1) {
			return false;
		}
		if (n > m) {
			return oneEditAway(b, a);
		}
		int i = 0, j = 0, cnt = 0;
		while (i < n && j < m && cnt <= 1) {
			char c1 = a.charAt(i), c2 = b.charAt(j);
			if (c1 == c2) {
				i++;
				j++;
			} else {
				if (n == m) {
					i++;
					j++;
					cnt++;
				} else {
					j++;
					cnt++;
				}
			}
		}
		return cnt <= 1;
	}

	// 交替合并字符串
	public String mergeAlternately(String s1, String s2) {
		int n = s1.length(), m = s2.length(), i = 0, j = 0;
		StringBuilder sb = new StringBuilder();
		while (i < n || j < m) {
			if (i < n) {
				sb.append(s1.charAt(i++));
			}
			if (j < m) {
				sb.append(s2.charAt(j++));
			}
		}
		return sb.toString();
	}

	// 链表相加
	public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
		ListNode ans = new ListNode(-1), temp = ans;
		int t = 0;
		while (l1 != null || l2 != null || t != 0) {
			if (l1 != null) {
				t += l1.val;
				l1 = l1.next;
			}
			if (l2 != null) {
				t += l2.val;
				l2 = l2.next;
			}
			temp.next = new ListNode(t % 10);
			temp = temp.next;
			t /= 10;
		}
		return ans.next;
	}

	// 分割数组
	public int partitionDisjoint(int[] nums) {
		int n = nums.length;
		int[] min = new int[n + 10];
		min[n - 1] = nums[n - 1];
		for (int i = n - 2; i >= 0; i--) {
			min[i] = Math.min(min[i + 1], nums[i]);
		}
		for (int i = 0, max = 0; i < n - 1; i++) {
			max = Math.max(max, nums[i]);
			if (max <= min[i + 1]) {
				return i + 1;
			}
		}
		return -1;
	}

	// 将数字翻译成字符串]
	public int translateNum(int num) {
		String s = String.valueOf(num);
		int a = 1, b = 1;
		for (int i = 2; i <= s.length(); i++) {
			String tmp = s.substring(i - 2, i);
			int c = tmp.compareTo("10") >= 0 && tmp.compareTo("25") <= 0 ? a + b : a;
			b = a;
			a = c;
		}
		return a;
	}
}