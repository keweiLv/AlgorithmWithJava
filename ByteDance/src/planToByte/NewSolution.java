package planToByte;

import java.util.*;

/**
 * @author Kezi
 * @date 2023年02月03日 23:07
 */
public class NewSolution {

    // 二叉树着色游戏
    private int x, lsz, rsz;

    public boolean btreeGameWinningMove(TreeNode root, int n, int x) {
        this.x = x;
        dfs(root);
        return Math.max(Math.max(lsz, rsz), n - 1 - lsz - rsz) * 2 > n;
    }

    private int dfs(TreeNode node) {
        if (node == null) {
            return 0;
        }
        int ls = dfs(node.left);
        int rs = dfs(node.right);
        if (node.val == x) {
            lsz = ls;
            rsz = rs;
        }
        return ls + rs + 1;
    }

    // 你能构造出连续值的最大数目
    public int getMaximumConsecutive(int[] coins) {
        int m = 0;
        Arrays.sort(coins);
        for (int coin : coins) {
            if (coin > m + 1) {
                break;
            }
            m += coin;
        }
        return m + 1;
    }

    // 最长连续序列
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        int longestLen = 0;
        for (int num : set) {
            if (!set.contains(num - 1)) {
                int curNum = num + 1;
                int curLen = 1;
                while (set.contains(curNum + 1)) {
                    curLen++;
                    curNum++;
                }
                longestLen = Math.max(longestLen, curLen);
            }
        }
        return longestLen;
    }

    // 有效的括号
    private static final Map<Character, Character> map = new HashMap<Character, Character>() {{
        put('{', '}');
        put('[', ']');
        put('(', ')');
        put('?', '?');
    }};

    public boolean isValid(String s) {
        if (s.length() > 0 && !map.containsKey(s.charAt(0))) {
            return false;
        }
        LinkedList<Character> stack = new LinkedList<Character>() {{
            add('?');
        }};
        for (Character c : s.toCharArray()) {
            if (map.containsKey(c)) {
                stack.addLast(c);
            } else if (!map.get(stack.removeLast()).equals(c)) {
                return false;
            }
        }
        return stack.size() == 1;
    }

    // 爬楼梯
    public int climbStairs(int n) {
        if (n <= 2) {
            return n;
        }
        int[] f = new int[n + 1];
        f[1] = 1;
        f[2] = 2;
        for (int i = 3; i <= n; i++) {
            f[i] = f[i - 1] + f[i - 2];
        }
        return f[n];
    }

    // 计算布尔二叉树的值
    public boolean evaluateTree(TreeNode root) {
        if (root.left == null) {
            return root.val == 1;
        }
        boolean l = evaluateTree(root.left);
        boolean r = evaluateTree(root.right);
        return root.val == 2 ? l || r : l && r;
    }

    // 警告一小时内使用相同员工卡大于等于三次的人
    public List<String> alertNames(String[] keyName, String[] keyTime) {
        Map<String, List<Integer>> timeMap = new HashMap<>();
        for (int i = 0; i < keyName.length; i++) {
            String name = keyName[i];
            String time = keyTime[i];
            timeMap.putIfAbsent(name, new ArrayList<>());
            int cnt = Integer.parseInt(time.substring(0, 2)) * 60 + Integer.parseInt(time.substring(3));
            timeMap.get(name).add(cnt);
        }
        List<String> res = new ArrayList<>();
        Set<String> strings = timeMap.keySet();
        for (String name : strings) {
            List<Integer> list = timeMap.get(name);
            Collections.sort(list);
            for (int i = 0; i < list.size(); i++) {
                int time1 = list.get(i - 2), time2 = list.get(i);
                int diff = time2 - time1;
                if (diff > 60) {
                    res.add(name);
                    break;
                }
            }
        }
        Collections.sort(res);
        return res;
    }

    // 删除子文件夹
    public List<String> removeSubfolders(String[] folder) {
        Arrays.sort(folder);
        List<String> res = new ArrayList<>();
        res.add(folder[0]);
        for (int i = 1; i < folder.length; i++) {
            int m = res.get(res.size() - 1).length();
            int n = folder[i].length();
            if (m >= n || !(res.get(res.size() - 1).equals(folder[i].substring(0, m)) && folder[i].charAt(m) == '/')) {
                res.add(folder[i]);
            }
        }
        return res;
    }

    // 数组能形成多少数对
    public int[] numberOfPairs(int[] nums) {
        int[] cnt = new int[101];
        for (int x : nums) {
            cnt[x]++;
        }
        int s = 0;
        for (int v : cnt) {
            s += v / 2;
        }
        return new int[]{s, nums.length - s * 2};
    }

    // 装满杯子需要的最短总时长
    public int fillCups(int[] amount) {
        Arrays.sort(amount);
        int a = amount[0], b = amount[1], c = amount[2];
        if (a + b <= c) {
            return c;
        } else {
            int t = a + b - c;
            return t % 2 == 0 ? t / 2 + c : t / 2 + c + 1;
        }
    }

    // 最大平均通过率
    public double maxAverageRatio(int[][] classes, int extraStudents) {
        PriorityQueue<double[]> pq = new PriorityQueue<>((a, b) -> {
            double x = (a[0] + 1) / (a[1] + 1) - a[0] / a[1];
            double y = (b[0] + 1) / (b[1] + 1) - b[0] / b[1];
            return Double.compare(y, x);
        });
        for (int[] item : classes) {
            pq.offer(new double[]{item[0], item[1]});
        }
        while (extraStudents-- > 0) {
            double[] poll = pq.poll();
            double a = poll[0] + 1, b = poll[1] + 1;
            pq.offer(new double[]{a, b});
        }
        double ans = 0;
        while (!pq.isEmpty()) {
            double[] poll = pq.poll();
            ans += poll[0] / poll[1];
        }
        return ans / classes.length;
    }

    // 最好的扑克手牌
    public String bestHand(int[] ranks, char[] suits) {
        boolean flush = true;
        for (int i = 1; i < 5 && flush; i++) {
            flush = suits[i] == suits[i - 1];
        }
        if (flush) {
            return "FLUSH";
        }
        int[] cnt = new int[14];
        boolean pair = false;
        for (int x : ranks) {
            if (++cnt[x] == 3) {
                return "Three of a Kind";
            }
            pair = pair || cnt[x] == 2;
        }
        return pair ? "Pair" : "High Card";
    }

    // 石子游戏Ⅱ
    public int stoneGameII(int[] piles) {
        int n = piles.length, sum = 0;
        int[][] dp = new int[n][n + 1];
        for (int i = n - 1; i >= 0; i--) {
            sum += piles[i];
            for (int M = 1; M <= n; M++) {
                if (i + 2 * M >= n) {
                    dp[i][M] = sum;
                } else {
                    for (int x = 1; x <= 2 * M; x++) {
                        dp[i][M] = Math.max(dp[i][M], sum - dp[i + x][Math.max(M, x)]);
                    }
                }
            }
        }
        return dp[0][1];
    }

    // 循环码排列
    public List<Integer> circularPermutation(int n, int start) {
        List<Integer> ans = new ArrayList<>();
        for (int i = 0; i < 1 << n; i++) {
            ans.add(i ^ (i >> 1) ^ start);
        }
        return ans;
    }

    // 使数组中所有元素都等于零
    public int minimumOperations(int[] nums) {
        boolean[] set = new boolean[101];
        set[0] = true;
        int ans = 0;
        for (int x : nums) {
            if (!set[x]) {
                ans++;
                set[x] = true;
            }
        }
        return ans;
    }
}