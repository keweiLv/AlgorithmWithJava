package newStart;

/**
 * 实现一个魔法字典
 */
public class MagicDictionary {

    class Trie {
        boolean isEnd;
        Trie[] child;

        public Trie() {
            isEnd = false;
            child = new Trie[26];
        }
    }

    Trie root;

    public MagicDictionary() {
        root = new Trie();
    }

    public void buildDict(String[] dictionary) {
        for (String word : dictionary) {
            Trie cur = root;
            for (int i = 0; i < word.length(); i++) {
                int idx = word.charAt(i) - 'a';
                if (cur.child[idx] == null) {
                    cur.child[idx] = new Trie();
                }
                cur = cur.child[idx];
            }
            cur.isEnd = true;
        }
    }

    public boolean search(String searchWord) {
        return dfs(searchWord, root, 0, false);
    }

    private boolean dfs(String searchWord, Trie root, int idx, boolean modified) {
        if (idx == searchWord.length()) {
            return modified && root.isEnd;
        }
        int cur = searchWord.charAt(idx) - 'a';
        if (root.child[cur] != null) {
            if (dfs(searchWord, root.child[cur], idx + 1, modified)) {
                return true;
            }
        }
        if (!modified) {
            for (int i = 0; i < 26; i++) {
                if (i != cur && root.child[i] != null) {
                    if (dfs(searchWord, root.child[i], idx + 1, true)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
}
