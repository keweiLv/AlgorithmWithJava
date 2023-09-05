package start230901;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * // 序列化和反序列化二叉搜索数
 */
public class Codec {

    private int i;
    private List<String> nums;
    private final int INF = 1 << 30;

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        nums = new ArrayList<>();
        dfs(root);
        return String.join(" ", nums);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data == null || "".equals(data)) {
            return null;
        }
        i = 0;
        nums = Arrays.asList(data.split(" "));
        return dfs(-INF, INF);
    }

    private void dfs(TreeNode root) {
        if (root == null) {
            return;
        }
        nums.add(String.valueOf(root.val));
        dfs(root.left);
        dfs(root.right);
    }

    private TreeNode dfs(int mi, int mx) {
        if (i == nums.size()) {
            return null;
        }
        int x = Integer.parseInt(nums.get(i));
        if (x < mi || x > mx) {
            return null;
        }
        TreeNode root = new TreeNode(x);
        i++;
        root.left = dfs(mi, x);
        root.right = dfs(x, mx);
        return root;
    }

}
