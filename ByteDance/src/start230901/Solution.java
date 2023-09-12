package start230901;

import java.util.*;

public class Solution {

    // 买铅笔和钢笔的方案数
    public long waysToBuyPensPencils(int total, int cost1, int cost2) {
        if (cost1 < cost2) {
            return waysToBuyPensPencils(total, cost2, cost1);
        }
        long res = 0, choose = 0;
        while (choose * cost1 <= total) {
            res += (total - choose * cost1) / cost2 + 1;
            choose++;
        }
        return res;
    }

    // 消灭怪物的最大数量
    public int eliminateMaximum(int[] dist, int[] speed) {
        int n = dist.length;
        int[] arrivalTimes = new int[n];
        for (int i = 0; i < n; i++) {
            arrivalTimes[i] = (dist[i] - 1) / speed[i] + 1;
        }
        Arrays.sort(arrivalTimes);
        for (int i = 0; i < n; i++) {
            if (arrivalTimes[i] <= i) {
                return i;
            }
        }
        return n;
    }

    // 从两个数字数组里生成最小数字
    public int minNumber(int[] nums1, int[] nums2) {
        int ans = 100;
        for (int a : nums1) {
            for (int b : nums2) {
                if (a == b) {
                    ans = Math.min(ans, a);
                } else {
                    ans = Math.min(ans, Math.min(a * 10 + b, b * 10 + a));
                }
            }
        }
        return ans;
    }

    // 最多可以摧毁的敌人堡垒数目
    public int captureForts(int[] forts) {
        int n = forts.length;
        int ans = 0, pre = -1;
        for (int i = 0; i < n; i++) {
            if (forts[i] == 1 || forts[i] == -1) {
                if (pre >= 0 && forts[i] != forts[pre]) {
                    ans = Math.max(ans, i - pre - 1);
                }
                pre = i;
            }
        }
        return ans;
    }

    // 最深叶节点的最近公共祖先
    private TreeNode treeNodeAns;
    private int maxDepth = -1;

    public TreeNode lcaDeepestLeaves(TreeNode root) {
        dfs(root, 0);
        return treeNodeAns;
    }

    private int dfs(TreeNode root, int depth) {
        if (root == null) {
            maxDepth = Math.max(maxDepth, depth);
            return depth;
        }
        int leftDepth = dfs(root.left, depth + 1);
        int rightDepth = dfs(root.right, depth + 1);
        if (leftDepth == rightDepth && leftDepth == maxDepth) {
            treeNodeAns = root;
        }
        return Math.max(leftDepth, rightDepth);
    }

    // 修车的最少时间
    public long repairCars(int[] ranks, int cars) {
        long l = 0, r = 1L * ranks[0] * cars * cars;
        while (l < r) {
            long m = l + r >> 1;
            if (check(ranks, cars, m)) {
                r = m;
            } else {
                l = m + 1;
            }
        }
        return l;
    }

    private boolean check(int[] ranks, int cars, long m) {
        long cnt = 0;
        for (int x : ranks) {
            cnt += (long) Math.sqrt(m / x);
        }
        return cnt >= cars;
    }

    // 每个小孩最多能分到多少糖果
    public int maximumCandies(int[] candies, long k) {
        int max = Arrays.stream(candies).max().getAsInt();
        int left = 0, right = max;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (getKids(candies, mid) < k) {
                right = mid - 1;
            } else {
                left = mid;
            }
        }
        return left;
    }

    private long getKids(int[] candies, int mid) {
        long cnt = 0;
        for (int candy : candies) {
            cnt += candy / mid;
        }
        return cnt;
    }

    // 计算列车到站时间
    public int findDelayedArrivalTime(int arrivalTime, int delayedTime) {
        return (arrivalTime + delayedTime) % 24;
    }

    // 最长回文串
    public int longestPalindrome(String s) {
        int[] arr = new int[128];
        for (char c : s.toCharArray()) {
            arr[c]++;
        }
        int count = 0;
        for (int i : arr) {
            count += (i % 2);
        }
        return count == 0 ? s.length() : (s.length() - count + 1);
    }

    // 多个数组求交集
    public List<Integer> intersection(int[][] nums) {
        int[] cnt = new int[10001];
        int n = nums.length;
        for (int[] num : nums) {
            for (int x : num) {
                cnt[x]++;
            }
        }
        List<Integer> ans = new ArrayList<>();
        for (int i = 0; i <= 1000; i++) {
            if (cnt[i] == n) {
                ans.add(i);
            }
        }
        return ans;
    }

    // 课程表
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        if (numCourses <= 0) {
            return new int[0];
        }
        HashSet<Integer>[] adj = new HashSet[numCourses];
        for (int i = 0; i < numCourses; i++) {
            adj[i] = new HashSet<>();
        }
        int[] inDegree = new int[numCourses];
        for (int[] p : prerequisites) {
            adj[p[1]].add(p[0]);
            inDegree[p[0]]++;
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }
        int[] res = new int[numCourses];
        int count = 0;
        while (!queue.isEmpty()) {
            Integer poll = queue.poll();
            res[count++] = poll;
            Set<Integer> tmp = adj[poll];
            for (Integer num : tmp) {
                inDegree[num]--;
                if (inDegree[num] == 0) {
                    queue.offer(num);
                }
            }
        }
        if (count == numCourses) {
            return res;
        }
        return new int[0];
    }

    // 最大子数组和
    public int maxSubArray(int[] nums) {
        int ans = nums[0], pre = nums[0];
        for (int i = 1; i < nums.length; i++) {
            pre = Math.max(pre + nums[i], nums[i]);
            ans = Math.max(ans, pre);
        }
        return ans;
    }

    // 删除链表的倒数第 N 个节点
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head);
        ListNode first = head;
        ListNode second = dummy;
        for (int i = 0; i < n; i++) {
            first = first.next;
        }
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        second.next = second.next.next;
        return dummy.next;
    }

    // 课程表四
    public List<Boolean> checkIfPrerequisite(int numCourses, int[][] prerequisites, int[][] queries) {
        boolean[][] f = new boolean[numCourses][numCourses];
        List<Integer>[] g = new List[numCourses];
        int[] indeg = new int[numCourses];
        Arrays.setAll(g, i -> new ArrayList<>());
        for (int[] p : prerequisites) {
            g[p[0]].add(p[1]);
            ++indeg[p[1]];
        }
        Deque<Integer> deque = new ArrayDeque<>();
        for (int i = 0; i < numCourses; i++) {
            if (indeg[i] == 0) {
                deque.offer(i);
            }
        }
        while (!deque.isEmpty()) {
            Integer i = deque.poll();
            for (int j : g[i]) {
                f[i][j] = true;
                for (int h = 0; h < numCourses; h++) {
                    f[h][j] |= f[h][i];
                }
                if (--indeg[j] == 0) {
                    deque.offer(j);
                }
            }
        }
        List<Boolean> ans = new ArrayList<>();
        for (int[] q : queries) {
            ans.add(f[q[0]][q[1]]);
        }
        return ans;
    }

    // 子集二
    List<List<Integer>> ans;
    List<Integer> path;

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        ans = new ArrayList<>();
        path = new ArrayList<>();
        Arrays.sort(nums);
        int n = nums.length;
        boolean[] vis = new boolean[n];
        backtrace(nums, 0, vis, 0);
        return ans;
    }

    private void backtrace(int[] nums, int start, boolean[] vis, int n) {
        ans.add(new ArrayList<>(path));
        for (int i = start; i < n; i++) {
            if (i > 0 && nums[i - 1] == nums[i] && !vis[i - 1]) {
                continue;
            }
            vis[i] = true;
            path.add(nums[i]);
            backtrace(nums, i + 1, vis, n);
            vis[i] = false;
            path.remove(path.size() - 1);
        }
    }
}


