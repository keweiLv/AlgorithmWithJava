package LC_StudyPlan75;


import java.util.*;

/**
 * @author Kezi
 * @date 2022年07月26日 22:32
 */
public class Solution {

	// 一维数组的动态和
	public int[] runningSum(int[] nums) {
		for (int i = 1; i < nums.length; i++) {
			nums[i] = nums[i - 1] + nums[i];
		}
		return nums;
	}

	// 寻找数组的中心索引
	public int pivotIndex(int[] nums) {
		int totel = Arrays.stream(nums).sum();
		int sum = 0;
		for (int i = 0; i < nums.length; i++) {
			if (2 * sum + nums[i] == totel) {
				return i;
			}
			sum += nums[i];
		}
		return -1;
	}

	// 同构字符串
	public boolean isIsomorphic(String s, String t) {
		Map<Character, Character> s2t = new HashMap<>(), t2s = new HashMap<>();
		for (int i = 0; i < s.length(); i++) {
			char a = s.charAt(i), b = t.charAt(i);
			if (s2t.containsKey(a) && s2t.get(a) != b || t2s.containsKey(b) && t2s.get(b) != a) {
				return false;
			}
			s2t.put(a, b);
			t2s.put(b, a);
		}
		return true;
	}

	// 判断子序列
	public boolean isSubsequence(String s, String t) {
		int n = s.length(), m = t.length();
		int i = 0, j = 0;
		while (i < n && j < m) {
			if (s.charAt(i) == t.charAt(j)) {
				i++;
			}
			j++;
		}
		return i == n;
	}

	// 合并两个有序链表
	public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
		if (l1 == null) {
			return l2;
		} else if (l2 == null) {
			return l1;
		} else if (l1.val < l2.val) {
			l1.next = mergeTwoLists(l1.next, l2);
			return l1;
		} else {
			l2.next = mergeTwoLists(l1, l2.next);
			return l2;
		}
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

	// 买卖股票的最佳时机
	public int maxProfit(int prices[]) {
		int minPrice = Integer.MAX_VALUE;
		int maxPrice = 0;
		for (int i = 0; i < prices.length; i++) {
			if (prices[i] < minPrice) {
				minPrice = prices[i];
			} else if (prices[i] - minPrice > maxPrice) {
				maxPrice = prices[i] - minPrice;
			}
		}
		return maxPrice;
	}

	// 最长回文串
	public int longestPalindrome(String s) {
		int[] count = new int[128];
		int len = s.length();
		for (int i = 0; i < len; i++) {
			char c = s.charAt(i);
			count[c]++;
		}
		int ans = 0;
		for (int v : count) {
			ans += v / 2 * 2;
			if (v % 2 == 1 && ans % 2 == 0) {
				ans++;
			}
		}
		return ans;
	}

	// N叉树的前序遍历
	public List<Integer> preorder(Node root) {
		List<Integer> res = new ArrayList<>();
		helper(root, res);
		return res;
	}

	private void helper(Node root, List<Integer> res) {
		if (root == null) {
			return;
		}
		res.add(root.val);
		for (Node ch : root.children) {
			helper(ch, res);
		}
	}

	// 二叉树的层序遍历
	public List<List<Integer>> levelOrder(TreeNode root) {
		List<List<Integer>> ans = new ArrayList<>();
		if (root == null) {
			return ans;
		}
		Queue<TreeNode> queue = new LinkedList<>();
		queue.offer(root);
		while (!queue.isEmpty()) {
			int len = queue.size();
			List<Integer> list = new ArrayList<>();
			for (int i = 0; i < len; i++) {
				TreeNode node = queue.poll();
				list.add(node.val);
				if (node.left != null) {
					queue.offer(node.left);
				}
				if (node.right != null) {
					queue.offer(node.right);
				}
			}
			ans.add(list);
		}
		return ans;
	}

	// 二分查找
	public int search(int[] nums, int target) {
		int low = 0, high = nums.length - 1;
		while (low <= high) {
			int mid = low + (high - low) / 2;
			int num = nums[mid];
			if (num > target) {
				high = mid - 1;
			} else if (num < target) {
				low = mid + 1;
			} else {
				return mid;
			}
		}
		return -1;
	}

	// 验证二查搜索树
	// 中序遍历
	public boolean isValidBST(TreeNode root) {
		Deque<TreeNode> deque = new LinkedList<>();
		double inorder = -Double.MAX_VALUE;
		while (!deque.isEmpty() || root != null) {
			while (root != null) {
				deque.push(root);
				root = root.left;
			}
			root = deque.pop();
			if (root.val <= inorder) {
				return false;
			}
			inorder = root.val;
			root = root.right;
		}
		return true;
	}

	// 二叉树的最近公共祖先
	public TreeNode lowestCommonAncestor(TreeNode cur, TreeNode p, TreeNode q) {
		if (cur == null || cur == p || cur == q) {
			return cur;
		}
		TreeNode left = lowestCommonAncestor(cur.left, p, q);
		TreeNode right = lowestCommonAncestor(cur.right, p, q);
		if (left == null) {
			return right;
		}
		if (right == null) {
			return left;
		}
		return cur;
	}

}
