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
    public List<List<Integer>> findLeaves(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<>();
        while (root != null) {
            List<Integer> list = new ArrayList<>();
            root = recur(root, list);
            ans.add(list);
        }
        return ans;
    }

    private TreeNode recur(TreeNode root, List<Integer> list) {
        if (root == null) {
            return null;
        }
        if (root.left == null && root.right == null) {
            list.add(root.val);
            return null;
        }
        root.left = recur(root.left, list);
        root.right = recur(root.right, list);
        return root;
    }

    // 检查一个数是否在数组中占绝大多数
    public boolean isMajorityElement(int[] nums, int target) {
        int n = nums.length;
        int left = 0, right = n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left + n / 2 < n && nums[left + n / 2] == target;
    }

    // 删除字符使频率相同
    public boolean equalFrequency(String word) {
        int[] set = new int[26];
        for (int i = 0; i < word.length(); i++) {
            set[word.charAt(i) - 'a']++;
        }
        for (int i = 0; i < 26; i++) {
            if (set[i] > 0) {
                set[i]--;
                int mark = 0;
                boolean ok = true;
                for (int num : set) {
                    if (num == 0) {
                        continue;
                    }
                    if (mark > 0 && num != mark) {
                        ok = false;
                        break;
                    }
                    mark = num;
                }
                if (ok) {
                    return true;
                }
                set[i]++;
            }
        }
        return false;
    }

    // 移动石子直到连续
    public int[] numMovesStones(int a, int b, int c) {
        int[] tmp = new int[]{a, b, c};
        Arrays.sort(tmp);
        a = tmp[0];
        b = tmp[1];
        c = tmp[2];
        return new int[]{
                c - a == 2 ? 0 : b - a <= 2 || c - b <= 2 ? 1 : 2, c - a - 2
        };
    }

    // 通知所有员工所需的时间
    public int numOfMinutes(int n, int headID, int[] manager, int[] informTime) {
        List<Integer> g[] = new ArrayList[n];
        Arrays.setAll(g, e -> new ArrayList<>());
        for (int i = 0; i < n; i++) {
            if (manager[i] >= 0) {
                g[manager[i]].add(i);
            }
        }
        return dfs(g, informTime, headID);
    }

    private int dfs(List<Integer>[] g, int[] informTime, int id) {
        int maxTime = 0;
        for (int num : g[id]) {
            maxTime = Math.max(maxTime, dfs(g, informTime, num));
        }
        return maxTime + informTime[id];
    }

    // 强整数
    public List<Integer> powerfulIntegers(int x, int y, int bound) {
        Set<Integer> set = new HashSet<>();
        for (int a = 1; a <= bound; a *= x) {
            for (int b = 1; a + b <= bound; b *= y) {
                set.add(a + b);
                if (y == 1) {
                    break;
                }
            }
            if (x == 1) {
                break;
            }
        }
        return new ArrayList<>(set);
    }

    // 检查替换后的词是否有效
    public static boolean isValid(String s) {
        String tmp = s;
        while (!tmp.isBlank()) {
            String replace = tmp.replace("abc", "");
            if (replace.equals(tmp)) {
                return false;
            }
            tmp = replace;
        }
        return true;
    }

    // 摘水果
    public int maxTotalFruits(int[][] fruits, int startPos, int k) {
        int left = lowerBound(fruits, startPos - k);
        int right = left, s = 0, n = fruits.length;
        for (; right < n && fruits[right][0] <= startPos; right++) {
            s += fruits[right][1];
        }
        int ans = s;
        for (; right < n && fruits[right][0] <= startPos + k; right++) {
            s += fruits[right][1];
            while (fruits[right][0] * 2 - fruits[left][0] - startPos > k && fruits[right][0] - fruits[left][0] * 2 + startPos > k) {
                s -= fruits[left++][1];
            }
            ans = Math.max(ans, s);
        }
        return ans;
    }

    private int lowerBound(int[][] fruits, int target) {
        int left = -1, right = fruits.length;
        while (left + 1 < right) {
            int mid = left + (right - left) / 2;
            if (fruits[mid][0] < target) {
                left = mid;
            } else {
                right = mid;
            }
        }
        return right;
    }

    // 有序数组中的缺失元素
    public int missingElement(int[] nums, int k) {
        int lose = nums[nums.length - 1] - nums[0] + 1 - nums.length;
        if (k > lose) {
            return nums[nums.length - 1] + k - lose;
        }
        int l = 0;
        int r = nums.length - 1;
        while (l < r - 1) {
            int mid = l + (r - l) >> 1;
            int loseCnt = nums[mid] - nums[l] - (mid - l);
            if (k > loseCnt) {
                l = mid;
                k -= loseCnt;
            } else {
                r = mid;
            }
        }
        return nums[l] + k;
    }

    // 处理用时最长的那个任务的员工
    public int hardestWorker(int n, int[][] logs) {
        int ans = 0;
        int last = 0, mx = 0;
        for (int[] log : logs) {
            int uid = log[0], time = log[1];
            time -= last;
            if (time > mx || (time == mx && uid < ans)) {
                ans = uid;
                mx = time;
            }
            last += time;
        }
        return ans;
    }

    // 找出变味映射
    public int[] anagramMappings(int[] nums1, int[] nums2) {
        Map<Integer, Integer> D = new HashMap<>();
        for (int i = 0; i < nums2.length; i++) {
            D.put(nums2[i], i);
        }
        int[] ans = new int[nums1.length];
        int t = 0;
        for (int x : nums1) {
            ans[t++] = D.get(x);
        }
        return ans;
    }

    // 数青蛙
    private static final char[] pre = new char['s'];

    static {
        char[] charArray = "croakc".toCharArray();
        for (int i = 1; i < charArray.length; i++) {
            pre[charArray[i]] = charArray[i - 1];
        }
    }

    public int minNumberOfFrogs(String croakOfFrogs) {
        int[] cnt = new int['s'];
        for (char ch : croakOfFrogs.toCharArray()) {
            char c = pre[ch];
            if (cnt[c] > 0) {
                cnt[c]--;
            } else if (ch != 'c') {
                return -1;
            }
            cnt[ch]++;
        }
        if (cnt['c'] > 0 || cnt['r'] > 0 || cnt['o'] > 0 || cnt['a'] > 0) {
            return -1;
        }
        return cnt['k'];
    }

    // 单线程CPU
    public int[] getOrder(int[][] tasks) {
        int n = tasks.length;
        int[][] nt = new int[n][3];
        for (int i = 0; i < n; i++) {
            nt[i] = new int[]{tasks[i][0], tasks[i][1], i};
        }
        Arrays.sort(nt, (a, b) -> a[0] - b[0]);
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> {
            if (a[1] != b[1]) {
                return a[1] - b[1];
            }
            return a[2] - b[2];
        });
        int[] ans = new int[n];
        for (int time = 1, j = 0, idx = 0; idx < n; ) {
            while (j < n && nt[j][0] <= time) {
                pq.add(nt[j++]);
            }
            if (pq.isEmpty()) {
                time = nt[j][0];
            } else {
                int[] poll = pq.poll();
                ans[idx++] = poll[2];
                time += poll[1];
            }
        }
        return ans;
    }

    // 总持续时间可被60整除的歌曲
    public int numPairsDivisibleBy60(int[] time) {
        int ans = 0;
        int[] cnt = new int[60];
        for (int t : time) {
            ans += cnt[(60 - t % 60) % 60];
            cnt[t % 60]++;
        }
        return ans;
    }

    // 雇佣K名工人的最低成本
    public double mincostToHireWorkers(int[] qs, int[] ws, int k) {
        int n = qs.length;
        double[][] ds = new double[n][2];
        for (int i = 0; i < n; i++) {
            ds[i][0] = ws[i] * 1.0 / qs[i];
            ds[i][1] = i * 1.0;
        }
        Arrays.sort(ds, (a, b) -> Double.compare(a[0], b[0]));
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> b - a);
        double ans = 1e18;
        for (int i = 0, tol = 0; i < n; i++) {
            int cur = qs[(int) ds[i][1]];
            tol += cur;
            pq.add(cur);
            if (pq.size() > k) {
                tol -= pq.poll();
            }
            if (pq.size() == k) {
                ans = Math.min(ans, tol * ds[i][0]);
            }
        }
        return ans;
    }

    // 有效时间的数目
    public int countTime(String time) {
        return func(time.substring(0, 2), 24) * func(time.substring(3, 5), 60);
    }

    private int func(String time, int m) {
        int cnt = 0;
        for (int i = 0; i < m; i++) {
            boolean a = time.charAt(0) == '?' || time.charAt(0) - '0' == i / 10;
            boolean b = time.charAt(1) == '?' || time.charAt(1) - '0' == i % 10;
            cnt += a && b ? 1 : 0;
        }
        return cnt;
    }

    // 单词替换
    class Node {
        boolean isEnd;
        Node[] tns = new Node[26];
    }

    Node root = new Node();

    public String replaceWords(List<String> dictionary, String sentence) {
        for (String str : dictionary) {
            add(str);
        }
        StringBuilder sb = new StringBuilder();
        for (String str : sentence.split(" ")) {
            sb.append(query(str)).append(" ");
        }
        return sb.substring(0, sb.length() - 1);
    }

    private String query(String str) {
        Node p = root;
        for (int i = 0; i < str.length(); i++) {
            int u = str.charAt(i) - 'a';
            if (p.tns[u] == null) {
                break;
            }
            if (p.tns[u].isEnd) {
                return str.substring(0, i + 1);
            }
            p = p.tns[u];
        }
        return str;
    }

    private void add(String str) {
        Node p = root;
        for (int i = 0; i < str.length(); i++) {
            int u = str.charAt(i) - 'a';
            if (p.tns[u] == null) {
                p.tns[u] = new Node();
            }
            p = p.tns[u];
        }
        p.isEnd = true;
    }

    // 可以被K整除的最小整数
    public int smallestRepunitDivByK(int k) {
        Set<Integer> set = new HashSet<>();
        int x = 1 % k;
        while (x > 0 && set.add(x)) {
            x = (x * 10 + 1) % k;
        }
        return x > 0 ? -1 : set.size() + 1;
    }

    // 子串能表示从1到N数字的二进制串
    public boolean queryString(String s, int n) {
        for (int i = 1; i <= n; i++) {
            String binaryString = Integer.toBinaryString(i);
            if (!s.contains(binaryString)) {
                return false;
            }
        }
        return true;
    }

    // 与对应负数同时存在的最大正整数
    public int findMaxK(int[] nums) {
        int ans = -1;
        Set<Integer> set = new HashSet<>();
        for (int x : nums) {
            set.add(x);
        }
        for (int nu : set) {
            if (set.contains(-nu)) {
                ans = Math.max(ans, nu);
            }
        }
        return ans;
    }

    // 所有子集
    private final List<List<Integer>> ans = new ArrayList<>();
    private final List<Integer> path = new ArrayList<>();
    private int[] nums;

    public List<List<Integer>> subsets(int[] nums) {
        this.nums = nums;
        dfs(0);
        return ans;
    }

    private void dfs(int i) {
        if (i == nums.length) {
            ans.add(new ArrayList<>(path));
            return;
        }
        // 不选nums[i]
        dfs(i + 1);
        // 选nums[i]
        path.add(nums[i]);
        dfs(i + 1);
        path.remove(path.size() - 1);
    }

    // 距离相等的条形码
    public int[] rearrangeBarcodes(int[] barcodes) {
        int n = barcodes.length;
        Integer[] tmp = new Integer[n];
        int max = 0;
        for (int i = 0; i < n; i++) {
            tmp[i] = barcodes[i];
            max = Math.max(max, barcodes[i]);
        }
        int[] cnt = new int[max + 1];
        for (int x : barcodes) {
            cnt[x]++;
        }
        Arrays.sort(tmp, (a, b) -> cnt[a] == cnt[b] ? a - b : cnt[b] - cnt[a]);
        int[] ans = new int[n];
        for (int k = 0, j = 0; k < 2; k++) {
            for (int i = k; i < n; i += 2) {
                ans[i] = tmp[j++];
            }
        }
        return ans;
    }

    // 字符串轮转
    public boolean isFlipedString(String s1, String s2) {
        return s1.length() == s2.length() && (s1 + s1).contains(s2);
    }

    // 重复的DNA序列
    public List<String> findRepeatedDnaSequences(String s) {
        List<String> ans = new ArrayList<>();
        int n = s.length();
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i + 10 <= n; i++) {
            String cur = s.substring(i, i + 10);
            int cnt = map.getOrDefault(cur, 0);
            if (cnt == 1) {
                ans.add(cur);
            }
            map.put(cur, cnt + 1);
        }
        return ans;
    }

    // 按列翻转得到最大值等行数
    public int maxEqualRowsAfterFlips(int[][] matrix) {
        int ans = 0, n = matrix[0].length;
        Map<String, Integer> cnt = new HashMap<>(16);
        for (int[] row : matrix) {
            int[] r = new int[n];
            for (int i = 0; i < n; i++) {
                r[i] = (char) (row[0] ^ row[i]);
            }
            Integer merge = cnt.merge(Arrays.toString(r), 1, Integer::sum);
            ans = Math.max(ans, merge);
        }
        return ans;
    }
}