package newStart;

import java.util.*;

public class Solution {

    // 删除最短的子数组使剩余数组有序
    public int findLengthOfShortestSubarray(int[] arr) {
        int n = arr.length, right = n - 1;
        while (right > 0 && arr[right - 1] <= arr[right]) {
            right--;
        }
        if (right == 0) {
            return 0;
        }
        int ans = right;
        for (int left = 0; left == 0 || arr[left - 1] <= arr[left]; left++) {
            while (right < n && arr[right] < arr[left]) {
                right++;
            }
            ans = Math.min(ans, right - left - 1);
        }
        return ans;
    }

    // 气温变化趋势
    public int temperatureTrend(int[] temperatureA, int[] temperatureB) {
        int n = temperatureA.length;
        int ans = 0;
        int cur = 0;
        for (int i = 1; i < n; i++) {
            if (temperatureA[i - 1] < temperatureA[i] && temperatureB[i - 1] < temperatureB[i]) {
                cur++;
            } else if (temperatureA[i - 1] == temperatureA[i] && temperatureB[i - 1] == temperatureB[i]) {
                cur++;
            } else if (temperatureA[i - 1] > temperatureA[i] && temperatureB[i - 1] > temperatureB[i]) {
                cur++;
            } else {
                cur = 0;
            }
            ans = Math.max(ans, cur);
        }
        return ans;
    }

    // 采购方案
    public int purchasePlans(int[] nums, int target) {
        int mod = 1000000007;
        int ans = 0;
        Arrays.sort(nums);
        int left = 0, right = nums.length - 1;
        while (left < right) {
            if (nums[left] + nums[right] > target) {
                right--;
            } else {
                ans += right - left;
                left++;
            }
            ans %= mod;
        }
        return ans % mod;
    }

    // 重复叠加字符串匹配
    public int repeatedStringMatch(String a, String b) {
        StringBuilder sb = new StringBuilder();
        int ans = 0;
        while (sb.length() < b.length() && ++ans > 0) {
            sb.append(a);
        }
        sb.append(a);
        int idx = sb.indexOf(b);
        if (idx == -1) {
            return -1;
        }
        return idx + b.length() > a.length() * ans ? ans + 1 : ans;
    }

    // 找出字符串中第一个匹配项的下标
    public int strStr(String haystack, String needle) {
        int n = haystack.length(), m = needle.length();
        char[] s = haystack.toCharArray(), p = needle.toCharArray();
        for (int i = 0; i <= n - m; i++) {
            int a = i, b = 0;
            while (b < m && s[a] == p[b]) {
                a++;
                b++;
            }
            if (b == m) {
                return i;
            }
        }
        return -1;
    }

    // 统计只差一个字符的字串数目
    public int countSubstrings(String s, String t) {
        int ans = 0;
        int m = s.length(), n = t.length();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (s.charAt(i) != t.charAt(j)) {
                    int l = 0, r = 0;
                    while (i - l > 0 && j - l > 0 && s.charAt(i - l - 1) == t.charAt(j - l - 1)) {
                        ++l;
                    }
                    while (i + r + 1 < m && j + r + 1 < n && s.charAt(i + r + 1) == t.charAt(j + r + 1)) {
                        ++r;
                    }
                    ans += (1 + l) * (1 + r);
                }
            }
        }
        return ans;
    }

    // 最小展台数量
    public int minNumBooths(String[] demand) {
        int[] cnt = new int[26];
        int[] cur = new int[26];
        for (String s : demand) {
            Arrays.fill(cur, 0);
            for (int i = 0; i < s.length(); i++) {
                int id = (int) s.charAt(i) - 'a';
                cur[id]++;
            }
            for (int i = 0; i < 26; i++) {
                cnt[i] = Math.max(cnt[i], cur[i]);
            }
        }
        int ans = 0;
        for (int num : cnt) {
            ans += num;
        }
        return ans;
    }

    // 数组中重复的元素
    public List<Integer> findDuplicates(int[] nums) {
        List<Integer> ans = new ArrayList<>();
        for (int num : nums) {
            if (nums[Math.abs(num) - 1] < 0) {
                ans.add(Math.abs(num));
            } else {
                nums[Math.abs(num) - 1] *= -1;
            }
        }
        return ans;
    }

    // 缺失的第一个正数
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                swap(nums, i, nums[i] - 1);
            }
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return n + 1;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    // 装饰树
    public TreeNode expandBinaryTree(TreeNode root) {
        if (root == null || (root.left == null && root.right == null)) {
            return root;
        }
        TreeNode leftNode = root.left;
        if (leftNode != null) {
            root.left = new TreeNode(-1);
            root.left.left = expandBinaryTree(leftNode);
        }
        TreeNode righNode = root.right;
        if (righNode != null) {
            root.right = new TreeNode(-1);
            root.right.right = expandBinaryTree(righNode);
        }
        return root;
    }

    // 统计字典序元首字符串的数目
    public int countVowelStrings(int n) {
        int[] f = {1, 1, 1, 1, 1};
        for (int i = 0; i < n - 1; ++i) {
            int s = 0;
            for (int j = 0; j < 5; ++j) {
                s += f[j];
                f[j] = s;
            }
        }
        return Arrays.stream(f).sum();
    }

    // 两点之间不包含任何点的最宽垂直面积
    public int maxWidthOfVerticalArea(int[][] points) {
        Arrays.sort(points, (a, b) -> a[0] - b[0]);
        int ans = 0;
        for (int i = 0; i < points.length - 1; i++) {
            ans = Math.max(ans, points[i + 1][0] - points[i][0]);
        }
        return ans;
    }

    // 交通枢纽
    public int transportationHub(int[][] path) {
        int[] in = new int[1001];
        int[] out = new int[1001];
        Set<Integer> set = new HashSet<>();
        for (int[] item : path) {
            int x = item[0], y = item[1];
            out[x]++;
            in[y]++;
            set.add(x);
            set.add(y);
        }
        int cnt = set.size();
        for (int i = 0; i < 1001; i++) {
            if (in[i] == cnt - 1 && out[i] == 0) {
                return i;
            }
        }
        return -1;
    }

    // 乘积小于K的子数组
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        int n = nums.length, ans = 0;
        if (k <= 1) {
            return 0;
        }
        for (int i = 0, j = 0, cur = 1; i < n; i++) {
            cur *= nums[i];
            while (cur >= k) {
                cur /= nums[j++];
            }
            ans += i - j + 1;
        }
        return ans;
    }

    // 无重复字符的最长字串
    public int lengthOfLongestSubstring(String s) {
        if (s.length() == 0) {
            return 0;
        }
        Map<Character, Integer> map = new HashMap<>();
        int ans = 0;
        int left = 0;
        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))) {
                left = Math.max(left, map.get(s.charAt(i)) + 1);
            }
            map.put(s.charAt(i), i);
            ans = Math.max(ans, i - left + 1);
        }
        return ans;
    }

    // 算数三元组的数目
    public int arithmeticTriplets(int[] nums, int diff) {
        Set<Integer> set = new HashSet<>();
        int ans = 0;
        for (int num : nums) {
            set.add(num);
            if (set.contains(num - diff) && set.contains(num - 2 * diff)) {
                ans++;
            }
        }
        return ans;
    }

    // Nim游戏
    public boolean canWinNim(int n) {
        return n % 4 != 0;
    }

    // 我能赢吗
    int n, t;
    int[] memo = new int[1 << 20];

    public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        n = maxChoosableInteger;
        t = desiredTotal;
        if (maxChoosableInteger >= desiredTotal) {
            return true;
        }
        if (maxChoosableInteger * (maxChoosableInteger + 1) / 2 < desiredTotal) {
            return false;
        }
        return dfs(0, 0) == 1;
    }

    private int dfs(int state, int tol) {
        if (memo[state] != 0) {
            return memo[state];
        }
        for (int i = 0; i < n; i++) {
            if (((state >> i) & 1) == 1) {
                continue;
            }
            if (tol + i + 1 >= t) {
                return memo[state] = 1;
            }
            if (dfs(state | (1 << i), tol + i + 1) == -1) {
                return memo[state] = 1;
            }
        }
        return memo[state] = -1;
    }

    // 隐藏个人信息
    public String maskPII(String s) {
        if (Character.isLetter(s.charAt(0))) {
            s = s.toLowerCase();
            int i = s.indexOf("@");
            return s.charAt(0) + "*****" + s.substring(i - 1);
        }
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (Character.isDigit(c)) {
                sb.append(c);
            }
        }
        s = sb.toString();
        int cnt = s.length() - 10;
        String suf = "***-***-" + s.substring(s.length() - 4);
        StringBuilder check = new StringBuilder();
        check.append("+");
        for (int i = 0; i < cnt; i++) {
            check.append("*");
        }
        return cnt == 0 ? suf : check + "-" + suf;
    }


    // 多边形三角剖分的最低得分
    int[] v;
    int[][] socreMemo;

    public int minScoreTriangulation(int[] values) {
        v = values;
        int n = v.length;
        socreMemo = new int[n][n];
        for (int i = 0; i < n; i++) {
            Arrays.fill(socreMemo[i], -1);
        }
        return scoreDfs(0, n - 1);
    }

    private int scoreDfs(int i, int j) {
        if (i + 1 == j) {
            return 0;
        }
        if (socreMemo[i][j] != -1) {
            return socreMemo[i][j];
        }
        int res = Integer.MAX_VALUE;
        for (int k = i + 1; k < j; k++) {
            res = Math.min(res, scoreDfs(i, k) + scoreDfs(k, j) + v[i] * v[j] * v[k]);
        }
        return socreMemo[i][j] = res;
    }

    // 交换一次的先前排列
    public int[] prevPermOpt1(int[] arr) {
        int n = arr.length;
        for (int i = n - 1; i > 0; i++) {
            if (arr[i - 1] > arr[i]) {
                for (int j = n - 1; i > i - 1; j--) {
                    if (arr[j] < arr[i - 1] && arr[j - 1] != arr[j]) {
                        int t = arr[i - 1];
                        arr[i - 1] = arr[j];
                        arr[j] = t;
                        return arr;
                    }
                }
            }
        }
        return arr;
    }

    // 共因子的数目
    public int commonFactors(int a, int b) {
        int ans = 0;
        for (int i = 1; i <= Math.min(a, b); i++) {
            if (a % i == 0 && b % i == 0) {
                ans++;
            }
        }
        return ans;
    }

    // 负二进制转换
    public String baseNeg2(int n) {
        if (n == 0) {
            return "0";
        }
        int k = 1;
        StringBuilder sb = new StringBuilder();
        while (n != 0) {
            if (n % 2 != 0) {
                sb.append(1);
                n -= k;
            } else {
                sb.append(0);
            }
            k *= -1;
            n /= 2;
        }
        return sb.reverse().toString();
    }

    // 移动石子直接连续二
    public int[] numMovesStonesII(int[] s) {
        Arrays.sort(s);
        int n = s.length;
        int e1 = s[n - 2] - s[0] - n + 2;
        int e2 = s[n - 1] - s[1] - n + 2;
        int maxMove = Math.max(e1, e2);
        if (e1 == 0 || e2 == 0) {
            return new int[]{Math.min(2, maxMove), maxMove};
        }
        int maxCnt = 0, left = 0;
        for (int right = 0; right < n; ++right) {
            while (s[right] - s[left] + 1 > n) {
                ++left;
            }
            maxCnt = Math.max(maxCnt, right - left + 1);
        }
        return new int[]{n - maxCnt, maxMove};
    }

    // 数组列表中的最大值
    public int maxDistance(List<List<Integer>> list) {
        List<Integer> init = list.get(0);
        int res = 0, minVal = init.get(0), maxVal = init.get(init.size() - 1);
        int flag = 0;
        for (List<Integer> item : list) {
            if (flag == 0) {
                flag++;
                continue;
            }
            res = Math.max(res, Math.max(Math.abs(item.get(item.size() - 1) - minVal), Math.abs(maxVal - item.get(0))));
            minVal = Math.min(minVal, item.get(0));
            maxVal = Math.max(maxVal, item.get(item.size() - 1));
        }
        return res;
    }

    // 链表中的下一个更大节点
    public int[] nextLargerNodes(ListNode head) {
        int n = 0;
        for (ListNode cur = head; cur != null; cur = cur.next) {
            n++;
        }
        int[] ans = new int[n];
        Deque<Integer> deque = new ArrayDeque<>();
        int i = 0;
        for (ListNode cur = head; cur != null; cur = cur.next) {
            while (!deque.isEmpty() && ans[deque.peek()] < cur.val) {
                ans[deque.pop()] = cur.val;
            }
            deque.push(i);
            ans[i++] = cur.val;
        }
        for (Integer idx : deque) {
            ans[idx] = 0;
        }
        return ans;
    }

    // 字符串的左右移
    public String stringShift(String s, int[][] shift) {
        int n = s.length();
        int count = 0;
        for (int i = 0; i < shift.length; i++) {
            count = shift[i][1] % n;
            if (shift[i][0] == 0 && shift[i][1] > 0) {
                s = s.substring(count) + s.substring(0, count);
            } else {
                s = s.substring(n - count) + s.substring(0, n - count);
            }
        }
        return s;
    }

    // 句子相似性
    public boolean areSentencesSimilar(String[] sentence1, String[] sentence2, List<List<String>> similarPairs) {
        if (sentence1.length != sentence2.length) {
            return false;
        }
        Set<String> set = new HashSet<>();
        for (List<String> str : similarPairs) {
            set.add(str.get(0) + "#" + str.get(1));
        }
        for (int i = 0; i < sentence1.length; i++) {
            if (!sentence1[i].equals(sentence2[i]) && !set.contains(sentence1[i] + "#" + sentence2[i]) && !set.contains(sentence2[i] + "#" + sentence1[i])) {
                return false;
            }
        }
        return true;
    }

    // 困于环中的机器人
    public boolean isRobotBounded(String instructions) {
        int k = 0;
        int[] dist = new int[4];
        for (int i = 0; i < instructions.length(); i++) {
            char c = instructions.charAt(i);
            if (c == 'L') {
                k = (k + 1) % 4;
            } else if (c == 'R') {
                k = (k + 3) % 4;
            } else {
                dist[k]++;
            }
        }
        return (dist[0] == dist[2] && dist[1] == dist[3]) || (k != 0);
    }

    // 会议室
    public boolean canAttendMeetings(int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        });
        for (int i = 0; i < intervals.length - 1; i++) {
            if (intervals[i][1] > intervals[i + 1][0]) {
                return false;
            }
        }
        return true;
    }

    // 会议室二
    public int minMeetingRooms(int[][] intervals) {
        if (intervals == null || intervals.length == 0) {
            return 0;
        }
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        pq.add(intervals[0][1]);
        for (int i = 1; i < intervals.length; i++) {
            int last = pq.peek();
            if (last <= intervals[i][0]) {
                pq.poll();
                pq.add(intervals[i][1]);
            } else {
                pq.add(intervals[i][1]);
            }
        }
        return pq.size();
    }

    // 出现最频繁的偶数元素
    public int mostFrequentEven(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if (num % 2 == 0) {
                map.merge(num, 1, Integer::sum);
            }
        }
        int ans = -1, mx = 0;
        Set<Map.Entry<Integer, Integer>> entries = map.entrySet();
        for (Map.Entry<Integer, Integer> en : map.entrySet()) {
            int x = en.getKey(), v = en.getValue();
            if (v > mx || (v == mx && x < ans)) {
                ans = x;
                mx = v;
            }
        }
        return ans;
    }


    // 驼峰式匹配
    public List<Boolean> camelMatch(String[] queries, String pattern) {
        List<Boolean> ans = new ArrayList<>();
        for (String str : queries) {
            ans.add(check(str, pattern));
        }
        return ans;
    }

    private Boolean check(String str, String pattern) {
        int m = str.length(), n = pattern.length();
        int i = 0, j = 0;
        for (; j < n; i++, j++) {
            while (i < m && str.charAt(i) != pattern.charAt(j) && Character.isLowerCase(str.charAt(i))) {
                ++i;
            }
            if (i == m || str.charAt(i) != pattern.charAt(j)) {
                return false;
            }
        }
        while (i < m && Character.isLowerCase(str.charAt(i))) {
            ++i;
        }
        return i == m;
    }

    // 字符串解码
    public String decodeString(String s) {
        StringBuilder ans = new StringBuilder();
        int multi = 0;
        Deque<Integer> stackMulti = new ArrayDeque<>();
        Deque<String> stackStr = new ArrayDeque<>();
        for (Character c : s.toCharArray()) {
            if (c == '[') {
                stackMulti.addLast(multi);
                stackStr.add(ans.toString());
                multi = 0;
                ans = new StringBuilder();
            } else if (c == ']') {
                StringBuilder tmp = new StringBuilder();
                int curMulti = stackMulti.removeLast();
                for (int i = 0; i < curMulti; i++) {
                    tmp.append(ans);
                }
                ans = new StringBuilder(stackStr.removeLast() + tmp);
            } else if (c >= '0' && c <= '9') {
                multi = multi * 10 + Integer.parseInt(c + "");
            } else {
                ans.append(c);
            }
        }
        return ans.toString();
    }

    // 不领接植花
    public int[] gardenNoAdj(int n, int[][] paths) {
        List<Integer>[] g = new List[n];
        Arrays.setAll(g, x -> new ArrayList<>());
        for (int[] path : paths) {
            int x = path[0] - 1, y = path[1] - 1;
            g[x].add(y);
            g[y].add(x);
        }
        int[] ans = new int[n];
        boolean[] used = new boolean[5];
        for (int i = 0; i < n; i++) {
            Arrays.fill(used, false);
            for (int y : g[i]) {
                used[ans[y]] = true;
            }
            for (int c = 1; c < 5; c++) {
                if (!used[c]) {
                    ans[i] = c;
                    break;
                }
            }
        }
        return ans;
    }

    // 最接近的二叉搜索树值
    public int closestValue(TreeNode root, double target) {
        int l = root.val;
        int r = root.val;
        while (root != null) {
            if (target == root.val) {
                return root.val;
            } else if (target < root.val) {
                r = root.val;
                root = root.left;
            } else {
                l = root.val;
                root = root.right;
            }
        }
        return Math.abs(target - l) < Math.abs(target - r) ? l : r;
    }

    // 验证前序遍历序列二叉搜索树
    public boolean verifyPreorder(int[] preorder) {
        if (preorder == null || preorder.length == 0) {
            return true;
        }
        int n = preorder.length;
        int i = 1;
        for (; i < n; i++) {
            if (preorder[i] > preorder[0]) {
                break;
            }
        }
        for (int j = i; j < n; j++) {
            if (preorder[j] < preorder[0]) {
                return false;
            }
        }
        return verifyPreorder(Arrays.copyOfRange(preorder, 1, i)) && verifyPreorder(Arrays.copyOfRange(preorder, i, n));
    }

    /**
     * 摆动排序
     * 实际可用归纳法证明，假设【0，1，...k]已满足摆动排序，k+1不满足，记k-1,k,k+1分别为a,b,c
     * 1.若不满足降序，则 c > b,同时b >= a ,此时a<=b<c,交换b、c之后，变为a < c > b，满足条件
     * 2.若不满足升序，则c < b,同事 b <= a，此时c<b<=a,交换b、c之后，变为a > c < b,满足条件
     */
    public void wiggleSort(int[] nums) {
        boolean jud = true;
        for (int i = 0; i < nums.length - 1; i++) {
            if (jud) {
                if (nums[i] > nums[i + 1]) {
                    swap(nums, i, i + 1);
                }
            } else {
                if (nums[i] < nums[i + 1]) {
                    swap(nums, i, i + 1);
                }
            }
            jud = !jud;
        }
    }
}
