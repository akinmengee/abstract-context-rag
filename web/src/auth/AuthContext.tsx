import { createContext, useContext, useState, type ReactNode } from "react";

import { login as apiLogin, register as apiRegister } from "../api/client";

interface AuthContextValue {
  token: string | null;
  userEmail: string | null;
  login(email: string, password: string): Promise<void>;
  register(email: string, password: string): Promise<void>;
  logout(): void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem("acr_token"));
  const [userEmail, setUserEmail] = useState<string | null>(() => localStorage.getItem("acr_email"));

  function persist(newToken: string, email: string) {
    localStorage.setItem("acr_token", newToken);
    localStorage.setItem("acr_email", email);
    setToken(newToken);
    setUserEmail(email);
  }

  async function login(email: string, password: string) {
    const result = await apiLogin(email, password);
    persist(result.access_token, result.email);
  }

  async function register(email: string, password: string) {
    const result = await apiRegister(email, password);
    persist(result.access_token, result.email);
  }

  function logout() {
    localStorage.removeItem("acr_token");
    localStorage.removeItem("acr_email");
    setToken(null);
    setUserEmail(null);
  }

  return (
    <AuthContext.Provider value={{ token, userEmail, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth() must be used inside an AuthProvider");
  return value;
}
