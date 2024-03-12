package start240304;

import java.util.HashSet;
import java.util.Set;

/**
 * 在受污染的二叉树中查找元素
 */
public class FindElements {

    private final Set<Integer> set = new HashSet<>();

    public FindElements(TreeNode root) {
        dfs(root, 0);
    }

    private void dfs(TreeNode root, int i) {
        if (root == null) {
            return;
        }
        set.add(i);
        dfs(root.left, i * 2 + 1);
        dfs(root.right, i * 2 + 2);
    }

    public boolean find(int target) {
        return set.contains(target);
    }
}
