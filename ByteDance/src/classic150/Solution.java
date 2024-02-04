package classic150;

public class Solution {

    // 求根节点到叶节点数字之和
    public int sumNumbers(TreeNode root) {
        return preHelper(root, 0);
    }

    private int preHelper(TreeNode root, int i) {
        if (root == null) {
            return 0;
        }
        int temp = i * 10 + root.val;
        if (root.left == null && root.right == null) {
            return temp;
        }
        return preHelper(root.left, temp) + preHelper(root.right, temp);
    }
}
