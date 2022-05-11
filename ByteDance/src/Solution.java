import java.util.ArrayList;
import java.util.List;

/**
 * @Author: Kezi
 * @Date: 2022/5/11 21:37
 */
public class Solution {

    // 序列化与反序列化二叉搜索树
    public String serialize(TreeNode root) {
        if (root == null) {
            return null;
        }
        List<String> list = new ArrayList<>();
        dfs1(root, list);
        int n = list.size();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            sb.append(list.get(i));
            if (i != n - 1) {
                sb.append(",");
            }
        }
        return sb.toString();
    }

    private void dfs1(TreeNode root, List<String> list) {
        if (root == null) {
            return;
        }
        list.add(String.valueOf(root.val));
        dfs1(root.left, list);
        dfs1(root.right, list);
    }

    public TreeNode deserialize(String s) {
        if (s == null) {
            return null;
        }
        String[] ss = s.split(",");
        return dfs2(0, ss.length - 1, ss);
    }
    private TreeNode dfs2(int l, int r, String[] ss) {
        if (l > r) {
            return null;
        }
        int ll = l + 1, rr = r, t = Integer.parseInt(ss[l]);
        while (ll < rr) {
            int mid = ll + rr >> 1;
            if (Integer.parseInt(ss[mid]) > t) {
                rr = mid;
            } else {
                ll = mid + 1;
            }
        }
        if (Integer.parseInt(ss[rr]) <= t){
            rr++;
        }
        TreeNode ans = new TreeNode(t);
        ans.left = dfs2(l+1,rr-1,ss);
        ans.right = dfs2(rr,r,ss);
        return ans;
    }

}
