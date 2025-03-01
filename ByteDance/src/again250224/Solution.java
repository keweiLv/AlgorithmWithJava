package again250224;

import java.util.ArrayList;
import java.util.List;

public class Solution {

    /**
     * 分割回文串
     */
    private final List<List<String>> res = new ArrayList<>();
    private final List<String> path = new ArrayList<>();
    private String s;

    public List<List<String>> partition(String s) {
        this.s = s;
        dfs(0, 0);
        return res;
    }

    private void dfs(int i, int start) {
        if (i == s.length()) {
            System.out.println(path);
            res.add(new ArrayList<>(path));
            return;
        }
        if (i < s.length() - 1) {
            dfs(i + 1, start);
        }
        if (isPalindrome(start, i)) {
            path.add(s.substring(start, i + 1));
            dfs(i + 1, i + 1);
            path.remove(path.size() - 1);
        }
    }

    private boolean isPalindrome(int left, int right) {
        while (left < right) {
            if (s.charAt(left++) != s.charAt(right--)) {
                return false;
            }
        }
        return true;
    }

}
