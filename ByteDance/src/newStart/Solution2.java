package newStart;

import java.util.*;

public class Solution2 {

    // 两个非重叠子数组的最大和
    public int maxSumTwoNoOverlap(int[] nums, int firstLen, int secondLen) {
        int n = nums.length;
        int[] s = new int[n + 1];
        for (int i = 0; i < n; i++) {
            s[i + 1] = s[i] + nums[i];
        }
        return Math.max(f(s, firstLen, secondLen), f(s, secondLen, firstLen));
    }

    private int f(int[] s, int first, int sec) {
        int maxSumA = 0, res = 0;
        for (int i = first + sec; i < s.length; i++) {
            maxSumA = Math.max(maxSumA, s[i - sec] - s[i - sec - first]);
            res = Math.max(res, maxSumA + s[i] - s[i - sec]);
        }
        return res;
    }

    // 因子的组合
    public List<List<Integer>> getFactors(int n) {
        return dfs(2, n);
    }

    private List<List<Integer>> dfs(int start, int num) {
        if (num == 1) {
            return new ArrayList<>();
        }
        int qNum = (int) Math.sqrt(num);
        List<List<Integer>> result = new ArrayList<>();
        for (int i = start; i <= qNum; i++) {
            if (num % i == 0) {
                List<Integer> simple = new ArrayList<>();
                simple.add(i);
                simple.add(num / i);
                result.add(simple);
                List<List<Integer>> nexList = dfs(i, num / i);
                for (List<Integer> list : nexList) {
                    list.add(i);
                    result.add(list);
                }
            }
        }
        return result;
    }

    // 最长字符串链
    Map<String, Integer> ws = new HashMap<>();
    public int longestStrChain(String[] words) {
        for (String str : words) {
            ws.put(str, 0);
        }
        int ans = 0;
        for (String key : ws.keySet()) {
            ans = Math.max(ans, dfs(key));
        }
        return ans;
    }
    private int dfs(String s) {
        Integer cnt = ws.get(s);
        if (cnt > 0) {
            return cnt;
        }
        for (int i = 0; i < s.length(); i++) {
            String tmp = s.substring(0, i) + s.substring(i + 1);
            if (ws.containsKey(tmp)) {
                cnt = Math.max(cnt, dfs(tmp));
            }
        }
        ws.put(s, cnt + 1);
        return cnt + 1;
    }

    // 寻找二叉树的叶子节点
    public List<List<Integer>> findLeaves(TreeNode root){
        List<List<Integer>> ans = new ArrayList<>();
        while (root != null){
            List<Integer> list = new ArrayList<>();
            root = recur(root,list);
            ans.add(list);
        }
        return ans;
    }

    private TreeNode recur(TreeNode root, List<Integer> list) {
        if (root == null){
            return null;
        }
        if (root.left == null && root.right == null){
            list.add(root.val);
            return null;
        }
        root.left = recur(root.left,list);
        root.right = recur(root.right,list);
        return root;
    }
}
