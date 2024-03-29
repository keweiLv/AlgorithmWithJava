package AlgInterviewSummary;

import java.util.*;

/**
 * @author Kezi
 * @date 2022年10月27日 23:01
 * @describe leetcode算法面试题汇总
 */
public class Solution {

    // 搜索二维矩阵Ⅱ
    public boolean searchMatrix(int[][] matrix, int target) {
        int row = matrix.length - 1, col = 0;
        while (row >= 0 && col < matrix[0].length) {
            if (target > matrix[row][col]) {
                col++;
            } else if (target < matrix[row][col]) {
                row--;
            } else {
                return true;
            }
        }
        return false;
    }

    // 合并两个有序数组
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1;
        int j = n - 1;
        int end = m + n - 1;
        while (j >= 0) {
            nums1[end--] = (i >= 0 && nums1[i] > nums2[j]) ? nums1[i--] : nums2[j--];
        }
    }

    // 字母大小写全排列
    public List<String> letterCasePermutation(String S) {
        List<String> res = new ArrayList<>();
        char[] chars = S.toCharArray();
        dfs(chars, 0, res);
        return res;
    }

    private void dfs(char[] chars, int i, List<String> res) {
        if (i == chars.length) {
            res.add(new String(chars));
            return;
        }
        dfs(chars, i + 1, res);
        if (Character.isLetter(chars[i])) {
            chars[i] ^= 1 << 5;
            dfs(chars, i + 1, res);
        }
    }

    // 神奇字符串
    public int magicalString(int n) {
        char[] s = new char[n + 2];
        s[0] = 1;
        s[1] = s[2] = 2;
        char c = 2;
        for (int i = 2, j = 3; j < n; i++) {
            // 1^3 = 2,2^3 = 1
            c ^= 3;
            s[j++] = c;
            if (s[i] == 2) {
                s[j++] = c;
            }
        }
        int ans = 0;
        for (int i = 0; i < n; i++) {
            ans += 2 - s[i];
        }
        return ans;
    }

    // 滑动窗口的最大值
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        Deque<Integer> deque = new LinkedList<>();
        for (int i = 0; i < k; i++) {
            while (!deque.isEmpty() && nums[i] >= nums[deque.peekLast()]) {
                deque.pollLast();
            }
            deque.offerLast(i);
        }
        int[] ans = new int[n - k + 1];
        ans[0] = nums[deque.peekFirst()];
        for (int i = k; i < n; ++i) {
            while (!deque.isEmpty() && nums[i] >= nums[deque.peekLast()]) {
                deque.pollLast();
            }
            deque.offerLast(i);
            while (deque.peekFirst() <= i - k) {
                deque.pollFirst();
            }
            ans[i - k + 1] = nums[deque.peekFirst()];
        }
        return ans;
    }

    // 删除中间节点
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }

    // 有效数独
    public boolean isValidSudoku(char[][] board) {
        int[][] row = new int[9][10];
        int[][] col = new int[9][10];
        int[][] box = new int[9][10];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] == '.') {
                    continue;
                }
                int curNum = board[i][j] - '0';
                if (row[i][curNum] == 1) {
                    return false;
                }
                if (col[j][curNum] == 1) {
                    return false;
                }
                if (box[j / 3 + (i / 3) * 3][curNum] == 1) {
                    return false;
                }
                row[i][curNum] = 1;
                col[j][curNum] = 1;
                box[j / 2 + (i / 3) * 3][curNum] = 1;
            }
        }
        return true;
    }

    // 最大重复子字符串
    public int maxRepeating(String sequence, String word) {
        int n = sequence.length(), m = word.length(), ans = 0;
        int[] f = new int[n + 10];
        for (int i = 1; i <= n; i++) {
            if (i - m < 0) {
                continue;
            }
            if (sequence.substring(i - m, i).equals(word)) {
                f[i] = f[i - m] + 1;
            }
            ans = Math.max(ans, f[i]);
        }
        return ans;
    }

    // 设计goal解析器
    public String interpret(String s) {
        StringBuilder sb = new StringBuilder();
        int n = s.length();
        for (int i = 0; i < n; ) {
            if (s.charAt(i) == 'G') {
                sb.append('G');
                i++;
            } else if (i + 1 < n && s.charAt(i + 1) == ')') {
                sb.append('o');
                i += 2;
            } else {
                sb.append("al");
                i += 4;
            }
        }
        return sb.toString();
    }

    // 缺失的区间
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        List<String> ans = new ArrayList<>();
        int n = nums.length;
        if (n == 0) {
            ans.add(missHelper(lower - 1, upper + 1));
            return ans;
        }
        if (lower != nums[0]) {
            ans.add(missHelper(lower - 1, nums[0]));
        }
        for (int i = 0; i < n - 1; ++i) {
            if (nums[i] + 1 != nums[i + 1]) {
                ans.add(missHelper(nums[i], nums[i + 1]));
            }
        }
        if (upper != nums[n - 1]) {
            ans.add(missHelper(nums[n - 1], upper + 1));
        }
        return ans;
    }

    private String missHelper(int left, int right) {
        StringBuilder sb = new StringBuilder();
        if (left + 2 == right) {
            sb.append(left + 1);
        } else {
            sb.append(left + 1).append("->").append(right - 1);
        }
        return sb.toString();
    }

    // 模糊坐标
    public List<String> ambiguousCoordinates(String s) {
        List<String> result = new ArrayList<>();
        s = s.substring(1, s.length() - 1);
        for (int i = 1; i < s.length(); i++) {
            for (String x : getNums(s.substring(0, i))) {
                for (String y : getNums(s.substring(i))) {
                    result.add(new StringBuilder("(").append(x).append(", ").append(y).append(")").toString());
                }
            }
        }
        return result;
    }

    private List<String> getNums(String num) {
        List<String> list = new ArrayList<>();
        int i = 1;
        String left, right;
        while (i <= num.length()) {
            left = num.substring(0, i);
            right = num.substring(i++);
            if ((!left.equals("0") && left.charAt(0) == '0') || (!right.isEmpty() && right.charAt(right.length() - 1) == '0')) {
                continue;
            }
            if (right.isEmpty()) {
                list.add(left);
            } else {
                list.add(new StringBuilder(left).append(".").append(right).toString());
            }
        }
        return list;
    }

    // 统计一致字符串的数目
    public int countConsistentStrings(String allowed, String[] words) {
        boolean[] hash = new boolean[26];
        for (char c : allowed.toCharArray()) {
            hash[c - 'a'] = true;
        }
        int ans = 0;
        out:
        for (String s : words) {
            for (char c : s.toCharArray()) {
                if (!hash[c - 'a']) {
                    continue out;
                }
            }
            ans++;
        }
        return ans;
    }
}
