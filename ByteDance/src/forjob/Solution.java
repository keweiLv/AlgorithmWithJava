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
		dfs(nums,len,0,path,used,res);
		return res;
	}

	private void dfs(int[] nums, int len, int depth, Deque<Integer> path, boolean[] used, List<List<Integer>> res) {
		if (depth == len){
			res.add(new ArrayList<>(path));
			return;
		}
		for (int i = 0;i<len;i++){
			if (!used[i]){
				path.addLast(nums[i]);
				used[i] = true;
				dfs(nums,len,depth+1,path,used,res);
				used[i] = false;
				path.removeLast();
			}
		}
	}

}
