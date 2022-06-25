import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
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
        if (Integer.parseInt(ss[rr]) <= t) {
            rr++;
        }
        TreeNode ans = new TreeNode(t);
        ans.left = dfs2(l + 1, rr - 1, ss);
        ans.right = dfs2(rr, r, ss);
        return ans;
    }

    // 二查搜索树中第K小的元素
    public int kthSmallest(TreeNode root, int k) {
        Deque<TreeNode> stack = new ArrayDeque<>();
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                stack.addLast(root);
                root = root.left;
            }
            root = stack.pollLast();
            if (--k == 0) {
                return root.val;
            }
            root = root.right;
        }
        return -1;
    }

    // 单值二叉树
    int val = -1;

    public boolean isUnivalTree(TreeNode root) {
        if (val == -1) {
            val = root.val;
        }
        if (root == null) {
            return true;
        }
        if (root.val != val) {
            return false;
        }
        return isUnivalTree(root.left) && isUnivalTree(root.right);
    }


    //有序数组的平方
    public int[] sortedSquares(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        for (int i = 0, j = n - 1, pos = n - 1; i <= j; ) {
            if (nums[i] * nums[i] > nums[j] * nums[j]) {
                ans[pos] = nums[i] * nums[i];
                ++i;
            } else {
                ans[pos] = nums[j] * nums[j];
                --j;
            }
            --pos;
        }
        return ans;
    }

    // 单词距离
    public int findClosest(String[] ws, String a, String b) {
        int n = ws.length, ans = n;
        for (int i = 0, p = -1, q = -1; i < n; i++) {
            String t = ws[i];
            if (t.equals(a)) {
                p = i;
            }
            if (t.equals(b)) {
                q = i;
            }
            if (p != -1 && q != -1) {
                ans = Math.min(ans, Math.abs(p - q));
            }
        }
        return ans;
    }

    // 两数之和Ⅱ
    public int[] twoSum(int[] numbers, int target) {
        int i = 0;
        int j = numbers.length - 1;
        while (i < j) {
            int sum = numbers[i] + numbers[j];
            if (sum < target) {
                i++;
            } else if (sum > target) {
                j--;
            } else {
                return new int[]{i + 1, j + 1};
            }
        }
        return new int[]{-1,-1};
    }
}