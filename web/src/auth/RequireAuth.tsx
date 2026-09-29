import type { ReactElement } from "react";
import { Navigate } from "react-router-dom";

import { useAuth } from "./AuthContext";

export function RequireAuth({ children }: { children: ReactElement }): ReactElement {
  const { token } = useAuth();
  return token ? children : <Navigate to="/login" replace />;
}
