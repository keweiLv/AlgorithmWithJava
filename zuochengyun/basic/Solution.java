package zuochengyun.basic;

import java.util.HashMap;
import java.util.Map;

/**
 * 左程云基础课
 */
public class Solution {

    // 环形链表二
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null || head.next.next == null) {
            return null;
        }
        ListNode slow = head.next;
        ListNode fast = head.next.next;
        while (slow != fast) {
            if (fast.next == null || fast.next.next == null) {
                return null;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        fast = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }

    // 二叉树的最大宽度
    Map<Integer, Integer> map = new HashMap<>();
    int ans = 0;

    public int widthOfBinaryTree(TreeNode root) {
        dfs(root, 1, 0);
        return ans;
    }

    private void dfs(TreeNode root, int u, int depth) {
        if (root == null) {
            return;
        }
        if (!map.containsKey(depth)) {
            map.put(depth, u);
        }
        ans = Math.max(ans, u - map.get(depth) + 1);
        u = u - map.get(depth) + 1;
        dfs(root.left, u << 1, depth + 1);
        dfs(root.right, u << 1 | 1, depth + 1);
    }

    // 验证二叉搜索树
    long pre = Long.MIN_VALUE;

    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return true;
        }
        if (!isValidBST(root.left)) {
            return false;
        }
        if (root.val <= pre) {
            return false;
        }
        pre = root.val;
        return isValidBST(root.right);
    }

    // 平衡二叉树
    public boolean isBalanced(TreeNode root) {
        if (root == null) {
            return true;
        }
        return Math.abs(depth(root.left) - depth(root.right)) < 2 && isBalanced(root.left) && isBalanced(root.right);
    }

    private int depth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return Math.max(depth(root.left), depth(root.right)) + 1;
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
}
